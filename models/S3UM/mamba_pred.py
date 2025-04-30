import torch
import os
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange
import math
from torchvision import transforms
from matplotlib.backends.backend_pdf import PdfPages

# Original positional embedding functions (unchanged)
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    embed_dim: output dimension for each position
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (with cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # 2 (grid_size, grid_size) arrays
    grid = np.stack(grid, axis=0)  # (2, grid_size, grid_size)
    grid = grid.reshape([2, 1, grid_size, grid_size])  # (2, 1, grid_size, grid_size)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)  # (grid_size*grid_size, embed_dim)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)  # (1+grid_size*grid_size, embed_dim)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class SSMKernel(nn.Module):
    """
    Selective State Space model kernel for sequence modeling
    with input-dependent parameters (A, B, C, delta)
    """

    def __init__(self, d_model, d_state, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Parameters for input-dependent A matrix
        self.A_proj = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_state)
        )

        # Parameters for input-dependent B matrix
        self.B_proj = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_state)
        )

        # Parameters for input-dependent C matrix
        self.C_proj = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_state)
        )

        # Direct term D (skip connection)
        self.D = nn.Parameter(torch.randn(d_model))

        # Parameter controlling discretization step size
        log_dt = torch.linspace(math.log(dt_min), math.log(dt_max), d_model)
        self.register_buffer('log_dt', log_dt)

        # Selective mechanism parameters
        self.delta_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, L, D]
        Returns: [B, L, D]
        """
        B, L, D = x.shape

        # Compute selective gating
        delta = self.delta_proj(x)  # [B, L, D]

        # Convert to feature dimension first
        x_features = rearrange(x, 'b l d -> b d l')  # [B, D, L]

        # Reshape input for parameter generation
        x_input = x.reshape(-1, D)  # [B*L, D]

        # Generate input-dependent A matrix for each position
        A_log = self.A_proj(x_input)  # [B*L, N]
        A_log = A_log.reshape(B, L, self.d_state)  # [B, L, N]
        # Make A negative to ensure stability
        A_log = -torch.abs(A_log)  # [B, L, N]

        # Generate input-dependent B matrix
        B_values = self.B_proj(x_input)  # [B*L, N]
        B_values = B_values.reshape(B, L, self.d_state)  # [B, L, N]

        # Generate input-dependent C matrix
        C_values = self.C_proj(x_input)  # [B*L, N]
        C_values = C_values.reshape(B, L, self.d_state)  # [B, L, N]

        # Get discrete parameters
        dt = torch.exp(self.log_dt)  # [D]

        # Ensure delta has right shape for selective scan
        delta = rearrange(delta, 'b l d -> b d l')  # [B, D, L]

        # Perform selective scan with all input-dependent parameters
        y = selective_ssm_scan_with_dynamic_A_B_C(
            x_features, A_log, B_values, C_values, delta, dt, self.D
        )

        # Convert back to sequence-first
        y = rearrange(y, 'b d l -> b l d')  # [B, L, D]

        return y


# Modified scan function to handle fully input-dependent parameters
def selective_ssm_scan_with_dynamic_A_B_C(u, A_log, B_values, C_values, delta, dt, D=None):
    """
    Selective state space model scan operation with input-dependent A, B, and C

    u: [B, D, L]
    A_log: [B, L, N] - input-dependent A matrix (in log space)
    B_values: [B, L, N] - input-dependent B matrix
    C_values: [B, L, N] - input-dependent C matrix
    delta: [B, D, L] - selective parameter
    dt: [D]
    D: [D] - optional direct term

    Returns: [B, D, L]
    """
    B_batch, D, L = u.shape
    N = B_values.shape[2]

    # Initialize state
    x = torch.zeros(B_batch, D, N, device=u.device)

    # For sequential scan with dynamic parameters
    ys = []
    for i in range(L):
        # Get current timestep parameters
        current_A = A_log[:, i, :]  # [B, N]
        current_B = B_values[:, i, :]  # [B, N]
        current_C = C_values[:, i, :]  # [B, N]

        # Create discretized A (dA) for each batch element
        dA = torch.exp(torch.einsum('bn,d->bdn', current_A, dt))  # [B, D, N]

        # Create discretized B (dB) for each batch element
        dB = torch.einsum('bn,d->bdn', current_B, dt)  # [B, D, N]

        # Apply selective update mechanism
        current_input = torch.einsum('bd,bdn->bdn', u[:, :, i], dB)
        x = delta[:, :, i:i + 1] * x + (1 - delta[:, :, i:i + 1]) * current_input # delta * h(t-1) + (1 - delta) * x

        # Apply state update with dynamic A
        x = x * dA

        # Compute output with dynamic C
        # Shape: [B, D, N] * [B, N] -> [B, D]
        y = torch.sum(x * current_C.unsqueeze(1), dim=2)  # [B, D]

        # Add skip connection if provided
        if D is not None:
            y = y + u[:, :, i] * D

        ys.append(y)

    # Stack outputs
    return torch.stack(ys, dim=2)  # [B, D, L]

# class S6Block(nn.Module):
#     """
#     Mamba-inspired Selective State Space Block
#     """
#
#     def __init__(self, dim, d_state=16, dropout=0.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#
#         # Selective SSM mechanism
#         self.ssm = nn.Sequential(
#             nn.Linear(dim, 2 * dim),
#             nn.SiLU(),
#             SSMKernel(2 * dim, d_state),
#             nn.Linear(2 * dim, dim),
#             nn.Dropout(dropout)
#         )
#
#         # FFN
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, 4 * dim),
#             nn.GELU(),
#             nn.Linear(4 * dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         # First sublayer: SSM with residual
#         x = x + self.ssm(self.norm1(x))
#
#         # Second sublayer: FFN with residual
#         x = x + self.ffn(self.norm2(x))
#
#         return x

# class S6Block(nn.Module):
#     """
#     Dual-branch Mamba-inspired Selective State Space Block
#     with multiplication between branch outputs
#     """
#
#     def __init__(self, dim, d_state=16, dropout=0.0):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         # self.norm2 = nn.LayerNorm(dim)
#
#         # Branch 1: Selective SSM mechanism
#         self.ssm_branch = nn.Sequential(
#             nn.Linear(dim, 2 * dim),
#             nn.SiLU(),
#             SSMKernel(2 * dim, d_state),
#             nn.Linear(2 * dim, dim),
#             nn.Dropout(dropout)
#         )
#
#         # Branch 2: FFN branch
#         self.ffn_branch = nn.Sequential(
#             nn.Linear(dim, 4 * dim),
#             nn.GELU(),
#             nn.Linear(4 * dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         # Normalize input
#         x_norm = self.norm(x)
#
#         # Process through both branches
#         ssm_output = self.ssm_branch(x_norm)
#         ffn_output = self.ffn_branch(x_norm)
#
#         # Multiply branch outputs and add residual connection
#         output = x + ssm_output * ffn_output
#
#         return output
class S6Block(nn.Module):
    """
    S6Block with Mamba-style architecture featuring selective SSM
    """

    def __init__(self, dim, d_state=16, d_conv=4, dropout=0.1):
        super().__init__()
        # 层标准化
        self.norm = nn.LayerNorm(dim)

        # 左分支 - SSM路径
        self.in_proj = nn.Linear(dim, dim)
        # 深度卷积层
        self.conv1d = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=dim
        )
        self.activation = nn.SiLU()
        self.ssm = SSMKernel(dim, d_state)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # 右分支 - 门控路径
        # self.gate_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # 保存残差连接
        residual = x

        # 层标准化
        x_norm = self.norm(x)

        # 左分支处理
        left = self.in_proj(x_norm)

        # 卷积处理 - 需要调整维度 [B, L, D] -> [B, D, L]
        left = left.transpose(1, 2)
        left = self.conv1d(left)[:, :, :-self.conv1d.padding[0]]  # 移除padding引入的额外部分
        left = left.transpose(1, 2)  # 恢复 [B, L, D]

        # 激活函数
        left = self.activation(left)

        # 右分支 - 计算门控值
        # gate = self.gate_proj(x_norm)
        gate = self.activation(x_norm)

        # SSM处理
        left = self.ssm(left)

        # 门控机制 - 左分支和右分支相乘
        y = left * gate

        # 投影并应用dropout
        y = self.out_proj(y)
        y = self.dropout(y)

        # 残差连接
        output = residual + y

        return output
class MaskingNet(nn.Module):
    """
    Masking network with Mamba-Enhanced Anatomical Structure Selection (S3UM)
    """

    def __init__(
            self,
            num_tokens,  # Number of image patches/tokens
            embed_dim=256,  # Embedding dimension
            depth=5,  # Number of S6 blocks
            d_state=16,  # State dimension for SSM
            dropout=0.0,  # Dropout rate
            norm_layer=nn.LayerNorm
    ):
        super().__init__()

        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim), requires_grad=False)

        # Mamba-based blocks replacing traditional Transformer blocks
        self.blocks = nn.ModuleList([
            S6Block(embed_dim, d_state=d_state, dropout=dropout)
            for _ in range(depth)
        ])

        # Final normalization and mask head
        self.norm = norm_layer(embed_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, num_tokens),
            nn.Sigmoid()
        )

        self.initialize_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Xavier initialization for linear layers
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_weights(self):
        # Initialize positional embeddings with sinusoidal pattern
        grid_size = int(np.sqrt(self.pos_embed.shape[1] - 1))
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize cls token with small random values
        torch.nn.init.normal_(self.cls_token, std=.02)

        # Initialize other weights
        self.apply(self._init_weights)

    def forward(self, x):
        """
        x: [B, N, D] tensor of patch tokens
        returns: [B, N] mask probabilities
        """
        # Add class token and positional embedding
        B, N, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # Apply Improved S6 blocks
        for blk in self.blocks:
            x = blk(x)

        # Extract features via global pooling (excluding cls token)
        x = self.norm(x)
        x = x[:, 1:].mean(dim=1)  # global pool without cls token

        # Generate mask probabilities
        mask_probs = self.mlp_head(x)

        return mask_probs

    def get_binary_mask(self, x, mask_ratio=0.75):
        """
        Generate binary mask by selecting top-k probabilities

        x: [B, N, D] tensor of patch tokens
        mask_ratio: fraction of patches to mask
        returns: [B, N] binary mask (1=keep, 0=mask)
        """
        mask_probs = self.forward(x)  # [B, N]

        # Select top-k indices (1-mask_ratio) to keep
        num_keep = int(mask_probs.shape[1] * (1 - mask_ratio))

        # Sort probabilities
        _, indices = torch.topk(mask_probs, num_keep, dim=1)

        # Create binary mask (0=masked, 1=kept)
        mask = torch.zeros_like(mask_probs)
        batch_indices = torch.arange(mask.shape[0]).unsqueeze(1).expand_as(indices)
        mask[batch_indices, indices] = 1.0

        return mask


# Test script
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Parameters
    batch_size = 2
    image_size = 224
    patch_size = 16
    num_patches = (image_size // patch_size) ** 2  # 196 patches
    embed_dim = 768


    # Function to load images from a directory
    def load_images_from_path(image_path, batch_size, image_size):
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load images from the directory
        images = []
        files = os.listdir(image_path)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Take batch_size images or all if there are fewer
        for i in range(min(batch_size, len(image_files))):
            img_file = os.path.join(image_path, image_files[i])
            img = Image.open(img_file).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)

        # If we have fewer images than batch_size, duplicate the last one
        while len(images) < batch_size:
            images.append(images[-1])

        # Stack all images into a batch
        return torch.stack(images)


    # Function to convert images to patch embeddings
    def create_image_patches(images, patch_size, embed_dim):
        B, C, H, W = images.shape
        P = patch_size

        # Reshape to patches
        patches = images.reshape(B, C, H // P, P, W // P, P).permute(0, 2, 4, 1, 3, 5)

        # Flatten patches: [B, H//P, W//P, C*P*P]
        patches = patches.reshape(B, (H // P) * (W // P), C * P * P)

        # Project to embedding dimension
        projection = nn.Linear(C * P * P, embed_dim).to(patches.device)
        patch_embeddings = projection(patches)

        return patch_embeddings


    # Path to your images - replace with your actual path
    image_path = "/archive/zhuchunzheng/dafenlei/midlate_500_resize/val/颅脑部分切面/"  # Replace with your actual path

    # Create model
    model = MaskingNet(
        num_tokens=num_patches,
        embed_dim=embed_dim,
        depth=3,
        d_state=16,
        dropout=0.1
    )

    # Load the pretrained model
    pretrained_path = "/archive/zhuchunzheng/tmi_CMUIM/output_dir_421/checkpoint-399.pth"
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    # If the checkpoint contains the full model
    if 'model' in checkpoint:
        # Extract just the model state
        model_state = checkpoint['model']
    elif isinstance(checkpoint, dict) and any(k.startswith('masking_net') for k in checkpoint.keys()):
        # The checkpoint already contains the model state
        model_state = checkpoint
    else:
        raise ValueError("Unexpected checkpoint format")

    # Create a new state dictionary with only the MaskingNet parameters
    masking_net_state = {}
    for key, value in model_state.items():
        # Check if the parameter belongs to the masking_net module
        if key.startswith('masking_net.'):
            # Remove the 'masking_net.' prefix to match your model's keys
            new_key = key[len('masking_net.'):]
            masking_net_state[new_key] = value

    # Load the parameters into your model
    # Use strict=False if your new model has a different structure
    model.load_state_dict(masking_net_state, strict=False)

    print("MaskingNet parameters loaded successfully!")

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load images and create patch embeddings
    images = load_images_from_path(image_path, batch_size, image_size).to(device)
    patch_embeddings = create_image_patches(images, patch_size, embed_dim)

    # Test forward pass
    print(f"Input shape: {patch_embeddings.shape}")
    mask_probs = model(patch_embeddings)
    print(f"Output mask probabilities shape: {mask_probs.shape}")
    # 创建保存PDF的文件夹
    output_dir = "/archive/zhuchunzheng/tmi_CMUIM/visualization_pdfs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取二进制掩码
    binary_mask = model.get_binary_mask(patch_embeddings, mask_ratio=0.75)
    print(f"Binary mask shape: {binary_mask.shape}")
    print(f"Number of kept tokens: {binary_mask.sum(dim=1)}")

    # Convert hex color #fef0e7 to RGB
    bg_color_hex = "#fef0e7"
    bg_color_rgb = [int(bg_color_hex[i:i+2], 16)/255 for i in (1, 3, 5)]  # Convert hex to RGB (0-1 range)

    # 为每个图像创建单独的可视化
    for i in range(batch_size):
        # 获取图像基本名称（用于文件名）
        base_name = f"image_{i}"

        # 1. 原始图像
        plt.figure(figsize=(6, 6), facecolor=bg_color_hex)
        # 转换图像从张量为显示格式
        img = images[i].detach().cpu().permute(1, 2, 0)
        # 反归一化
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)
        plt.imshow(img)
        plt.axis('off')  # 关闭坐标轴
        plt.tight_layout()
        # 设置背景颜色
        plt.gca().set_facecolor(bg_color_hex)
        # 保存为PDF，设置背景色
        with PdfPages(os.path.join(output_dir, f"{base_name}_origin.pdf")) as pdf:
            plt.savefig(pdf, format='pdf', facecolor=bg_color_hex, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 2. 掩码概率图 - 调整colorbar
        fig, ax = plt.subplots(figsize=(6, 6), facecolor=bg_color_hex)
        prob_map = mask_probs[i].detach().cpu().reshape(int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
        im = ax.imshow(prob_map)
        # 关闭坐标轴刻度
        ax.set_xticks([])
        ax.set_yticks([])
        # 设置背景颜色
        ax.set_facecolor(bg_color_hex)
        fig.patch.set_facecolor(bg_color_hex)

        # 创建一个较短的colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cax.set_facecolor(bg_color_hex)

        plt.tight_layout()
        # 保存为PDF和SVG，设置背景色
        plt.savefig(os.path.join(output_dir, f"{base_name}_probabilities.svg"), format='svg', facecolor=bg_color_hex, bbox_inches='tight', pad_inches=0.1)
        with PdfPages(os.path.join(output_dir, f"{base_name}_probabilities.pdf")) as pdf:
            plt.savefig(pdf, format='pdf', facecolor=bg_color_hex, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 3. 二进制掩码图 - 调整colorbar
        fig, ax = plt.subplots(figsize=(6, 6), facecolor=bg_color_hex)
        mask_reshaped = binary_mask[i].detach().cpu().reshape(int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
        im = ax.imshow(mask_reshaped)
        # 关闭坐标轴刻度
        ax.set_xticks([])
        ax.set_yticks([])
        # 设置背景颜色
        ax.set_facecolor(bg_color_hex)
        fig.patch.set_facecolor(bg_color_hex)

        # 创建一个较短的colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Masked', 'Kept'])
        cax.set_facecolor(bg_color_hex)

        plt.tight_layout()
        # 保存为PDF和SVG，设置背景色
        plt.savefig(os.path.join(output_dir, f"{base_name}_mask.svg"), format='svg', facecolor=bg_color_hex, bbox_inches='tight', pad_inches=0.1)
        with PdfPages(os.path.join(output_dir, f"{base_name}_mask.pdf")) as pdf:
            plt.savefig(pdf, format='pdf', facecolor=bg_color_hex, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 额外: 也保存一张被掩码的图像（可选）
        plt.figure(figsize=(6, 6), facecolor=bg_color_hex)
        # 复制原始图像
        masked_img = images[i].clone()
        # 将掩码上采样到图像大小
        mask_upsampled = torch.nn.functional.interpolate(
            mask_reshaped.unsqueeze(0).unsqueeze(0).float(),
            size=(image_size, image_size),
            mode='nearest'
        ).squeeze().bool()
        # 按通道应用掩码
        for c in range(3):
            masked_img[c][mask_upsampled] = 0.5  # 将掩码区域变灰

        masked_img = masked_img.detach().cpu().permute(1, 2, 0)
        # 反归一化
        masked_img = masked_img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        masked_img = torch.clamp(masked_img, 0, 1)
        plt.imshow(masked_img)
        plt.axis('off')  # 关闭坐标轴
        # 设置背景颜色
        plt.gca().set_facecolor(bg_color_hex)

        plt.tight_layout()
        # 保存为PDF和SVG，设置背景色
        plt.savefig(os.path.join(output_dir, f"{base_name}_masked.svg"), format='svg', facecolor=bg_color_hex, bbox_inches='tight', pad_inches=0.1)
        with PdfPages(os.path.join(output_dir, f"{base_name}_masked.pdf")) as pdf:
            plt.savefig(pdf, format='pdf', facecolor=bg_color_hex, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    print(f"单独的PDF文件已保存到 {output_dir} 目录")

    # 最终可视化保存
    plt.figure(figsize=(12, 6), facecolor=bg_color_hex)
    plt.tight_layout()
    plt.savefig('mask_visualization.png', facecolor=bg_color_hex, bbox_inches='tight', pad_inches=0.1)
    print("Visualization saved to mask_visualization.png")

    # Test training functionality
    # Create a dummy target mask
    target_mask = torch.bernoulli(torch.ones(batch_size, num_patches) * 0.5).to(device)

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Sample training loop
    print("\nRunning a sample training step...")
    model.train()
    optimizer.zero_grad()

    # Forward pass
    pred_mask = model(patch_embeddings)
    loss = criterion(pred_mask, target_mask)

    # Backward pass
    loss.backward()
    optimizer.step()

    print(f"Training loss: {loss.item()}")

    # Verify grad flow
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    print(f"Gradient norm: {total_norm}")

    print("Test completed successfully!")