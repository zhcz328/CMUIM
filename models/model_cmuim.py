"""
CMUIM model for continual self-supervised learning in ultrasound imaging.

This module implements the Continual Masked Ultrasound Image Modeling (CMUIM)
framework that integrates masked autoencoding with selective state space models
for anatomically-aware representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from masking_net import MaskingNet
from einops import repeat, rearrange
import numpy as np
from util.positional_embedding import get_2d_sincos_pos_embed, interpolate_pos_embed
from timm.models.vision_transformer import PatchEmbed


class Block(nn.Module):
    """
    Transformer block with self-attention and feed-forward network.

    This is a standard Transformer block used for both encoder and decoder.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop,
                                          bias=qkv_bias, batch_first=True)

        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        # Self-attention block
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path(attn_output)

        # MLP block
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):
    """
    MLP module with dropout.

    Simple multilayer perceptron with GELU activation and dropout.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """
    Drop paths (stochastic depth) per sample.

    When applied in main path, this layer drops the entire path with probability p.
    """

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class CMUIM(nn.Module):
    """
    Continual Masked Ultrasound Image Modeling with S3UM.

    This model implements the complete CMUIM architecture that combines:
    1. S3UM for anatomically-aware mask generation
    2. Vision Transformer backbone for feature encoding
    3. Decoder for masked image reconstruction
    4. Bi-level optimization for continual learning

    The architecture is specially designed for ultrasound images to enhance
    representation learning while mitigating catastrophic forgetting in
    continual learning scenarios.
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.,
            norm_layer=nn.LayerNorm,
            norm_pix_loss=False,
            # S3UM parameters
            masking_depth=5,
            d_state=16,
            d_conv=4,
            # Loss weights
            lambda_gauss=1.0,
            lambda_kl=0.1,
            lambda_diversity=0.2
    ):
        """
        Initialize the CMUIM model.

        Args:
            img_size (int): Input image size
            patch_size (int): Patch size for tokenization
            in_chans (int): Input channels
            embed_dim (int): Embedding dimension
            depth (int): Depth of the encoder
            num_heads (int): Number of attention heads
            decoder_embed_dim (int): Embedding dimension for decoder
            decoder_depth (int): Depth of the decoder
            decoder_num_heads (int): Number of attention heads in decoder
            mlp_ratio (float): MLP ratio
            norm_layer (nn.Module): Normalization layer
            norm_pix_loss (bool): Whether to normalize pixel values in loss
            masking_depth (int): Depth of the masking network
            d_state (int): State dimension for SSM
            d_conv (int): Kernel size for depth-wise convolution
            lambda_gauss (float): Weight for Gaussian loss
            lambda_kl (float): Weight for KL divergence loss
            lambda_diversity (float): Weight for diversity loss
        """
        super().__init__()

        # Save parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm_pix_loss = norm_pix_loss
        self.lambda_gauss = lambda_gauss
        self.lambda_kl = lambda_kl
        self.lambda_diversity = lambda_diversity

        # Training states
        self.backbone_frozen = False
        self.maskingnet_frozen = False
        self.previous_encoder = None

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Masking network (S3UM)
        self.masking_net = MaskingNet(
            num_tokens=num_patches,
            embed_dim=embed_dim,
            depth=masking_depth,
            d_state=d_state,
            d_conv=d_conv,
            dropout=0.1,
            norm_layer=norm_layer
        )

    def initialize_weights(self):
        """Initialize model weights with appropriate strategies."""
        # Initialize patch embedding
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize positional embeddings with sinusoidal patterns
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize cls token, mask token and other parameters
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.mask_token, std=.02)

        # Initialize other modules
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize individual module weights."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def freeze_backbone(self):
        """
        Freeze the backbone encoder-decoder parameters but keep masking network trainable.
        """
        # Freeze patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        # Freeze encoder parts
        for param in self.cls_token.parameters():
            param.requires_grad = False

        # The positional embedding is not trainable by default

        # Freeze encoder blocks
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        # Freeze encoder norm
        for param in self.norm.parameters():
            param.requires_grad = False

        # Freeze decoder parts
        for param in self.decoder_embed.parameters():
            param.requires_grad = False

        for param in self.mask_token.parameters():
            param.requires_grad = False

        # Freeze decoder blocks
        for block in self.decoder_blocks:
            for param in block.parameters():
                param.requires_grad = False

        # Freeze decoder norm and prediction head
        for param in self.decoder_norm.parameters():
            param.requires_grad = False

        for param in self.decoder_pred.parameters():
            param.requires_grad = False

        self.backbone_frozen = True

    def unfreeze_backbone(self):
        """
        Unfreeze the backbone encoder-decoder parameters.
        """
        # Unfreeze patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = True

        # Unfreeze encoder parts
        for param in self.cls_token.parameters():
            param.requires_grad = True

        # Unfreeze encoder blocks
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = True

        # Unfreeze encoder norm
        for param in self.norm.parameters():
            param.requires_grad = True

        # Unfreeze decoder parts
        for param in self.decoder_embed.parameters():
            param.requires_grad = True

        for param in self.mask_token.parameters():
            param.requires_grad = True

        # Unfreeze decoder blocks
        for block in self.decoder_blocks:
            for param in block.parameters():
                param.requires_grad = True

        # Unfreeze decoder norm and prediction head
        for param in self.decoder_norm.parameters():
            param.requires_grad = True

        for param in self.decoder_pred.parameters():
            param.requires_grad = True

        self.backbone_frozen = False

    def freeze_maskingnet(self):
        """
        Freeze the masking network parameters.
        """
        for param in self.masking_net.parameters():
            param.requires_grad = False

        self.maskingnet_frozen = True

    def unfreeze_maskingnet(self):
        """
        Unfreeze the masking network parameters.
        """
        for param in self.masking_net.parameters():
            param.requires_grad = True

        self.maskingnet_frozen = False

    def store_previous_encoder(self):
        """
        Store the current encoder as the previous encoder for continual learning.
        """
        self.previous_encoder = type(self)(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            depth=len(self.blocks),
            decoder_embed_dim=self.decoder_embed.out_features,
            d_state=self.masking_net.d_state if hasattr(self.masking_net, 'd_state') else 16,
            norm_pix_loss=self.norm_pix_loss
        ).to(self.cls_token.device)

        # Copy parameters from current model
        self.previous_encoder.load_state_dict(self.state_dict())

        # Freeze the previous encoder
        for param in self.previous_encoder.parameters():
            param.requires_grad = False

    def patchify(self, imgs):
        """
        Convert images to patches.

        Args:
            imgs (torch.Tensor): [B, 3, H, W] images

        Returns:
            patches (torch.Tensor): [B, N, P^2*C] patches
        """
        p = self.patch_size
        h = w = self.img_size // p
        x = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        return x

    def unpatchify(self, patches):
        """
        Convert patches back to images.

        Args:
            patches (torch.Tensor): [B, N, P^2*C] patches

        Returns:
            imgs (torch.Tensor): [B, 3, H, W] images
        """
        p = self.patch_size
        h = w = self.img_size // p
        return rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                         h=h, w=w, p1=p, p2=p)

    def forward_encoder(self, x, mask):
        """
        Forward pass through the encoder with masking.

        Args:
            x (torch.Tensor): [B, 3, H, W] input images
            mask (torch.Tensor): [B, N] binary mask (0=masked, 1=kept)

        Returns:
            x (torch.Tensor): [B, N_visible+1, D] encoded features
            mask (torch.Tensor): [B, N] binary mask (used for decoding)
        """
        # Embed patches
        x = self.patch_embed(x)
        B, N, D = x.shape

        # Add position embeddings
        # We need to slice the position embeddings to match num_patches
        pos_embed = self.pos_embed[:, 1:, :]
        x = x + pos_embed

        # Mask tokens
        mask_expand = mask.unsqueeze(-1).expand(-1, -1, D)
        x_visible = torch.masked_select(x, mask_expand.bool()).reshape(B, -1, D)

        # Add cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x_visible], dim=1)

        # Apply encoder blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask

    def forward_decoder(self, x, mask):
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): [B, N_visible+1, D] encoded features
            mask (torch.Tensor): [B, N] binary mask (0=masked, 1=kept)

        Returns:
            pred (torch.Tensor): [B, N, patch_size^2*in_chans] reconstructed patches
        """
        B = x.shape[0]
        N = self.patch_embed.num_patches

        # Embed tokens from encoder to decoder dimension
        x = self.decoder_embed(x)

        # Extract class token and visible tokens
        cls_token = x[:, :1, :]
        x_visible = x[:, 1:, :]

        # Get positions for masked and visible tokens
        visible_indices = torch.nonzero(mask, as_tuple=True)
        masked_indices = torch.nonzero(~mask, as_tuple=True)

        # Prepare decoder input
        decoder_input = torch.zeros(B, N + 1, x.shape[-1], device=x.device)

        # Add class token
        decoder_input[:, 0, :] = cls_token.squeeze(1)

        # Add visible tokens
        for b in range(B):
            visible_idx = visible_indices[1][visible_indices[0] == b]
            visible_count = len(visible_idx)
            decoder_input[b, 1:1 + visible_count, :] = x_visible[b, :visible_count, :]

        # Create and add mask tokens
        mask_token = self.mask_token.repeat(B, N, 1)

        # Add mask tokens
        for b in range(B):
            masked_idx = masked_indices[1][masked_indices[0] == b] + 1  # +1 for cls token offset
            decoder_input[b, masked_idx, :] = mask_token[b, :len(masked_idx), :]

        # Add positional embeddings
        decoder_input = decoder_input + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            decoder_input = blk(decoder_input)

        decoder_output = self.decoder_norm(decoder_input)

        # Predict patches (exclude cls token)
        pred = self.decoder_pred(decoder_output[:, 1:, :])

        return pred

    def forward_loss(self, imgs, pred, mask):
        """
        Compute the loss between original and reconstructed patches.

        Args:
            imgs (torch.Tensor): [B, 3, H, W] original images
            pred (torch.Tensor): [B, N, P^2*C] predicted patches
            mask (torch.Tensor): [B, N] binary mask (0=masked, 1=kept)

        Returns:
            loss (torch.Tensor): Reconstruction loss
            pred (torch.Tensor): Predicted patches
            mask (torch.Tensor): Binary mask
        """
        # Patchify original images
        target = self.patchify(imgs)

        # Normalize target if required
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        # Compute loss only on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N]

        # Only compute loss on masked patches
        loss = (loss * (~mask)).sum() / (~mask).sum()

        return loss, pred, mask

    def forward(self, x, mask_ratio=0.75, train_mask=False):
        """
        Forward pass of the CMUIM model.

        Args:
            x (torch.Tensor): [B, 3, H, W] input images
            mask_ratio (float): Fraction of patches to mask
            train_mask (bool): Whether to train the masking network

        Returns:
            loss (torch.Tensor): Reconstruction loss
            loss_area (torch.Tensor, optional): Area regularization loss (if train_mask=True)
            loss_kl (torch.Tensor, optional): KL divergence loss (if train_mask=True)
            loss_diversity (torch.Tensor, optional): Diversity regularization loss (if train_mask=True)
        """
        if train_mask:
            # Get patch embeddings
            x_embed = self.patch_embed(x)

            # Generate mask probabilities and calculate regularization losses
            mask_probs, loss_area, loss_diversity = self.masking_net(x_embed, train_mask=True)

            # Convert probabilities to binary mask
            mask = self.masking_net.get_binary_mask(x_embed, mask_ratio)

            # In S3UM module, KL divergence is not used but kept for interface consistency
            loss_kl = torch.tensor(0.0, device=x.device)

            # Forward pass with masked image modeling
            _, mask = self.forward_encoder(x, mask)
            pred = self.forward_decoder(_, mask)

            # Calculate reconstruction loss
            loss, _, _ = self.forward_loss(x, pred, mask)

            return loss, loss_area, loss_kl, loss_diversity
        else:
            # Get patch embeddings
            x_embed = self.patch_embed(x)

            # Generate mask using masking network
            mask = self.masking_net.get_binary_mask(x_embed, mask_ratio)

            # Encode and decode
            latent, mask = self.forward_encoder(x, mask)
            pred = self.forward_decoder(latent, mask)

            # Compute loss
            loss, _, _ = self.forward_loss(x, pred, mask)

            return loss, latent

    def forward_features(self, x):
        """
        Extract features for downstream tasks.

        Args:
            x (torch.Tensor): [B, 3, H, W] input images

        Returns:
            features (torch.Tensor): [B, D] features for downstream tasks
        """
        # Patch embedding
        x = self.patch_embed(x)

        # Add positional embeddings
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x + self.pos_embed[:, 1:, :]], dim=1)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Use CLS token as feature
        return x[:, 0]