"""
Test script for the CMUIM architecture.
This script demonstrates the usage of the MaskingNet model and its visualization.
"""

import torch
import os
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse

from models.masking_net import MaskingNet
from util.config import get_config, CMUIMConfig


class TestCMUIM:
    """
    Test class for the CMUIM architecture with visualization capabilities.
    """

    def __init__(self, config: CMUIMConfig = None):
        """
        Initialize the test class with configuration parameters.

        Args:
            config: Configuration object
        """
        # Load configuration if not provided
        if config is None:
            self.config = get_config()
        else:
            self.config = config

        # Set random seed for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Extract configuration parameters
        self.batch_size = self.config.training.batch_size
        self.image_size = self.config.data.image_size
        self.patch_size = self.config.data.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.embed_dim = self.config.model.embed_dim
        self.output_dir = self.config.paths.visualization_dir
        self.pretrained_path = self.config.paths.pretrained_model_path
        self.image_path = os.path.join(self.config.paths.data_dir, "val/颅脑部分切面/")

        # Background color for visualization
        self.bg_color_hex = self.config.visualization.background_color
        self.bg_color_rgb = [int(self.bg_color_hex[i:i+2], 16)/255 for i in (1, 3, 5)]

        # Create output directory if not exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Initialize model
        self.model = MaskingNet(
            num_tokens=self.num_patches,
            embed_dim=self.config.model.embed_dim,
            depth=self.config.model.depth,
            d_state=self.config.model.d_state,
            dropout=self.config.model.dropout
        )

        # Device configuration
        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')

        # Load pretrained model
        self._load_pretrained_model()

        # Move model to device
        self.model = self.model.to(self.device)

        print(f"Model initialized on {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

    def _load_pretrained_model(self):
        """
        Load pretrained model weights.
        """
        if not self.pretrained_path or not os.path.exists(self.pretrained_path):
            print("No pretrained model found. Using random initialization.")
            return

        print(f"Loading pretrained model from {self.pretrained_path}")
        checkpoint = torch.load(self.pretrained_path, map_location='cpu')

        # Extract model state based on checkpoint format
        if 'model' in checkpoint:
            model_state = checkpoint['model']
        elif isinstance(checkpoint, dict) and any(k.startswith('masking_net') for k in checkpoint.keys()):
            model_state = checkpoint
        else:
            raise ValueError("Unexpected checkpoint format")

        # Create a new state dictionary with only the MaskingNet parameters
        masking_net_state = {}
        for key, value in model_state.items():
            # Check if the parameter belongs to the masking_net module
            if key.startswith('masking_net.'):
                # Remove the 'masking_net.' prefix to match the model's keys
                new_key = key[len('masking_net.'):]
                masking_net_state[new_key] = value

        # Load parameters into the model
        self.model.load_state_dict(masking_net_state, strict=False)
        print("MaskingNet parameters loaded successfully!")

    def load_images(self):
        """
        Load images from the specified directory.

        Returns:
            images: [B, C, H, W] tensor of images
        """
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.data.mean,
                std=self.config.data.std
            )
        ])

        # Load images from the directory
        images = []
        files = os.listdir(self.image_path)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Take batch_size images or all if there are fewer
        for i in range(min(self.batch_size, len(image_files))):
            img_file = os.path.join(self.image_path, image_files[i])
            img = Image.open(img_file).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)

        # If we have fewer images than batch_size, duplicate the last one
        while len(images) < self.batch_size:
            images.append(images[-1] if images else torch.zeros_like(transform(Image.new('RGB', (self.image_size, self.image_size)))))

        # Stack all images into a batch
        return torch.stack(images).to(self.device)

    def create_patch_embeddings(self, images):
        """
        Convert images to patch embeddings.

        Args:
            images: [B, C, H, W] tensor of images

        Returns:
            patch_embeddings: [B, N, D] tensor of patch embeddings
        """
        B, C, H, W = images.shape
        P = self.patch_size

        # Reshape to patches
        patches = images.reshape(B, C, H // P, P, W // P, P).permute(0, 2, 4, 1, 3, 5)

        # Flatten patches: [B, H//P, W//P, C*P*P]
        patches = patches.reshape(B, (H // P) * (W // P), C * P * P)

        # Project to embedding dimension
        projection = nn.Linear(C * P * P, self.embed_dim).to(patches.device)
        patch_embeddings = projection(patches)

        return patch_embeddings

    def generate_masks(self, patch_embeddings):
        """
        Generate masks using the MaskingNet model.

        Args:
            patch_embeddings: [B, N, D] tensor of patch embeddings

        Returns:
            mask_probs: [B, N] tensor of mask probabilities
            binary_masks: [B, N] tensor of binary masks
        """
        # Set model to evaluation mode
        self.model.eval()

        # Forward pass to get mask probabilities
        with torch.no_grad():
            mask_probs = self.model(patch_embeddings)
            binary_masks = self.model.get_binary_mask(
                patch_embeddings,
                mask_ratio=self.config.training.mask_ratio
            )

        return mask_probs, binary_masks

    def create_visualizations(self, images, mask_probs, binary_masks):
        """
        Create visualizations of the original images, mask probabilities, and binary masks.

        Args:
            images: [B, C, H, W] tensor of images
            mask_probs: [B, N] tensor of mask probabilities
            binary_masks: [B, N] tensor of binary masks
        """
        # Process each image in the batch
        for i in range(images.shape[0]):
            # Get image base name (for file name)
            base_name = f"image_{i}"

            # 1. Original image
            plt.figure(figsize=self.config.visualization.figure_size, facecolor=self.bg_color_hex)
            # Convert image from tensor to display format
            img = images[i].detach().cpu().permute(1, 2, 0)
            # Denormalize
            img = img * torch.tensor(self.config.data.std) + torch.tensor(self.config.data.mean)
            img = torch.clamp(img, 0, 1)
            plt.imshow(img)
            plt.axis('off')  # Turn off axes
            plt.tight_layout()
            # Set background color
            plt.gca().set_facecolor(self.bg_color_hex)
            # Save as PDF with background color
            if self.config.visualization.create_pdf:
                with PdfPages(os.path.join(self.output_dir, f"{base_name}_origin.pdf")) as pdf:
                    plt.savefig(pdf, format='pdf', facecolor=self.bg_color_hex,
                               bbox_inches='tight', pad_inches=0.1, dpi=self.config.visualization.dpi)
            plt.close()

            # 2. Mask probability map
            fig, ax = plt.subplots(figsize=self.config.visualization.figure_size, facecolor=self.bg_color_hex)
            prob_map = mask_probs[i].detach().cpu().reshape(
                int(np.sqrt(self.num_patches)), int(np.sqrt(self.num_patches))
            )
            im = ax.imshow(prob_map, cmap=self.config.visualization.colormap)
            # Turn off axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
            # Set background color
            ax.set_facecolor(self.bg_color_hex)
            fig.patch.set_facecolor(self.bg_color_hex)

            # Create a shorter colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cax.set_facecolor(self.bg_color_hex)

            plt.tight_layout()
            # Save as PDF and SVG with background color
            if self.config.visualization.create_svg:
                plt.savefig(os.path.join(self.output_dir, f"{base_name}_probabilities.svg"),
                           format='svg', facecolor=self.bg_color_hex,
                           bbox_inches='tight', pad_inches=0.1, dpi=self.config.visualization.dpi)
            if self.config.visualization.create_pdf:
                with PdfPages(os.path.join(self.output_dir, f"{base_name}_probabilities.pdf")) as pdf:
                    plt.savefig(pdf, format='pdf', facecolor=self.bg_color_hex,
                               bbox_inches='tight', pad_inches=0.1, dpi=self.config.visualization.dpi)
            plt.close()

            # 3. Binary mask map
            fig, ax = plt.subplots(figsize=self.config.visualization.figure_size, facecolor=self.bg_color_hex)
            mask_reshaped = binary_masks[i].detach().cpu().reshape(
                int(np.sqrt(self.num_patches)), int(np.sqrt(self.num_patches))
            )
            im = ax.imshow(mask_reshaped, cmap='binary')
            # Turn off axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
            # Set background color
            ax.set_facecolor(self.bg_color_hex)
            fig.patch.set_facecolor(self.bg_color_hex)

            # Create a shorter colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Masked', 'Kept'])
            cax.set_facecolor(self.bg_color_hex)

            plt.tight_layout()
            # Save as PDF and SVG with background color
            if self.config.visualization.create_svg:
                plt.savefig(os.path.join(self.output_dir, f"{base_name}_mask.svg"),
                           format='svg', facecolor=self.bg_color_hex,
                           bbox_inches='tight', pad_inches=0.1, dpi=self.config.visualization.dpi)
            if self.config.visualization.create_pdf:
                with PdfPages(os.path.join(self.output_dir, f"{base_name}_mask.pdf")) as pdf:
                    plt.savefig(pdf, format='pdf', facecolor=self.bg_color_hex,
                               bbox_inches='tight', pad_inches=0.1, dpi=self.config.visualization.dpi)
            plt.close()

            # 4. Optional: Masked image
            plt.figure(figsize=self.config.visualization.figure_size, facecolor=self.bg_color_hex)
            # Copy original image
            masked_img = images[i].clone()
            # Upsample mask to image size
            mask_upsampled = torch.nn.functional.interpolate(
                mask_reshaped.unsqueeze(0).unsqueeze(0).float(),
                size=(self.image_size, self.image_size),
                mode='nearest'
            ).squeeze().bool()
            # Apply mask by channel
            for c in range(3):
                masked_img[c][mask_upsampled] = 0.5  # Make masked regions gray

            masked_img = masked_img.detach().cpu().permute(1, 2, 0)
            # Denormalize
            masked_img = masked_img * torch.tensor(self.config.data.std) + torch.tensor(self.config.data.mean)
            masked_img = torch.clamp(masked_img, 0, 1)
            plt.imshow(masked_img)
            plt.axis('off')  # Turn off axes
            # Set background color
            plt.gca().set_facecolor(self.bg_color_hex)

            plt.tight_layout()
            # Save as PDF and SVG with background color
            if self.config.visualization.create_svg:
                plt.savefig(os.path.join(self.output_dir, f"{base_name}_masked.svg"),
                           format='svg', facecolor=self.bg_color_hex,
                           bbox_inches='tight', pad_inches=0.1, dpi=self.config.visualization.dpi)
            if self.config.visualization.create_pdf:
                with PdfPages(os.path.join(self.output_dir, f"{base_name}_masked.pdf")) as pdf:
                    plt.savefig(pdf, format='pdf', facecolor=self.bg_color_hex,
                               bbox_inches='tight', pad_inches=0.1, dpi=self.config.visualization.dpi)
            plt.close()

        print(f"Visualizations saved to {self.output_dir}")

    def test_training_step(self):
        """
        Test a single training step with dummy data.
        """
        # Create dummy target mask
        target_mask = torch.bernoulli(torch.ones(self.batch_size, self.num_patches) * 0.5).to(self.device)

        # Load images and create patch embeddings
        images = self.load_images()
        patch_embeddings = self.create_patch_embeddings(images)

        # Set model to training mode
        self.model.train()

        # Define loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate)

        # Forward pass
        optimizer.zero_grad()
        pred_mask = self.model(patch_embeddings)
        loss = criterion(pred_mask, target_mask)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Verify gradient flow
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        print(f"Training loss: {loss.item()}")
        print(f"Gradient norm: {total_norm}")

    def run(self):
        """
        Run the complete test procedure.
        """
        print("Loading images...")
        images = self.load_images()

        print("Creating patch embeddings...")
        patch_embeddings = self.create_patch_embeddings(images)

        print("Generating masks...")
        mask_probs, binary_masks = self.generate_masks(patch_embeddings)

        if self.config.visualization.save_visualizations:
            print("Creating visualizations...")
            self.create_visualizations(images, mask_probs, binary_masks)

        print("Testing training step...")
        self.test_training_step()

        print("Test completed successfully!")


def parse_args():
    """
    Parse command line arguments.

    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Test CMUIM model.")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to configuration file.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for testing.")
    parser.add_argument("--image_size", type=int, default=None,
                        help="Size of input images.")
    parser.add_argument("--mask_ratio", type=float, default=None,
                        help="Fraction of patches to mask.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for saving visualizations.")
    parser.add_argument("--no_visualize", action="store_true",
                        help="Disable visualization.")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = get_config(args.config_path)

    # Override configuration with command line arguments
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.image_size is not None:
        config.data.image_size = args.image_size
    if args.mask_ratio is not None:
        config.training.mask_ratio = args.mask_ratio
    if args.output_dir is not None:
        config.paths.visualization_dir = args.output_dir
    if args.no_visualize:
        config.visualization.save_visualizations = False

    # Create test instance
    tester = TestCMUIM(config)

    # Run test
    tester.run()