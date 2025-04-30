"""
Visualization utilities for CMUIM architecture.

This module implements visualization functions for model outputs and masks.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_masks(images, masks, h, w, patch_size, output_path=None, bg_color="#fef0e7", dpi=300, fig_size=6):
    """
    Visualize original images, mask probabilities, and binary masks.

    Args:
        images (torch.Tensor): [B, 3, H, W] original images
        masks (torch.Tensor): [B, N] binary masks (1=kept, 0=masked)
        h (int): Height in patches
        w (int): Width in patches
        patch_size (int): Size of each patch
        output_path (str, optional): Path to save visualizations
        bg_color (str): Background color for visualizations
        dpi (int): DPI for saved figures
        fig_size (int): Figure size
    """
    # Denormalize images
    denorm_images = denormalize_images(images)

    # Reshape masks to 2D grid
    reshaped_masks = masks.reshape(-1, h, w)

    # Create masked versions of images
    masked_images = apply_masks_to_images(denorm_images, reshaped_masks, patch_size)

    # Convert RGB background color to array
    bg_color_rgb = [int(bg_color[i:i+2], 16)/255 for i in (1, 3, 5)]

    # Process each image in the batch
    for i in range(images.shape[0]):
        # Base name for output files
        base_name = f"image_{i}" if output_path else None

        # Create figure for original image
        plt.figure(figsize=(fig_size, fig_size), facecolor=bg_color)
        plt.imshow(denorm_images[i])
        plt.axis('off')
        plt.tight_layout()
        plt.gca().set_facecolor(bg_color)

        # Save figure if output path provided
        if output_path:
            output_file = os.path.join(output_path, f"{base_name}_original.pdf")
            with PdfPages(output_file) as pdf:
                plt.savefig(pdf, format='pdf', facecolor=bg_color,
                           bbox_inches='tight', pad_inches=0.1, dpi=dpi)
        plt.close()

        # Create figure for binary mask
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), facecolor=bg_color)
        im = ax.imshow(reshaped_masks[i], cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(bg_color)
        fig.patch.set_facecolor(bg_color)

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Masked', 'Kept'])
        cax.set_facecolor(bg_color)

        # Save figure if output path provided
        if output_path:
            output_file = os.path.join(output_path, f"{base_name}_mask.pdf")
            plt.savefig(output_file, format='pdf', facecolor=bg_color,
                       bbox_inches='tight', pad_inches=0.1, dpi=dpi)
            # Save SVG version
            svg_output = os.path.join(output_path, f"{base_name}_mask.svg")
            plt.savefig(svg_output, format='svg', facecolor=bg_color,
                       bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # Create figure for masked image
        plt.figure(figsize=(fig_size, fig_size), facecolor=bg_color)
        plt.imshow(masked_images[i])
        plt.axis('off')
        plt.gca().set_facecolor(bg_color)

        # Save figure if output path provided
        if output_path:
            output_file = os.path.join(output_path, f"{base_name}_masked.pdf")
            with PdfPages(output_file) as pdf:
                plt.savefig(pdf, format='pdf', facecolor=bg_color,
                           bbox_inches='tight', pad_inches=0.1, dpi=dpi)
            # Save SVG version
            svg_output = os.path.join(output_path, f"{base_name}_masked.svg")
            plt.savefig(svg_output, format='svg', facecolor=bg_color,
                       bbox_inches='tight', pad_inches=0.1)
        plt.close()

    if output_path:
        print(f"Visualizations saved to {output_path}")


def denormalize_images(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize images for visualization.

    Args:
        images (torch.Tensor): [B, 3, H, W] normalized images
        mean (list): Mean values for normalization
        std (list): Standard deviation values for normalization

    Returns:
        denorm_images (np.ndarray): [B, H, W, 3] denormalized images in numpy format
    """
    # Create mean and std tensors
    mean_tensor = torch.tensor(mean, device=images.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=images.device).view(1, 3, 1, 1)

    # Denormalize
    denorm_images = images * std_tensor + mean_tensor

    # Clamp values to [0, 1]
    denorm_images = torch.clamp(denorm_images, 0, 1)

    # Convert to numpy and change to HWC format
    denorm_images = denorm_images.detach().cpu().permute(0, 2, 3, 1).numpy()

    return denorm_images


def apply_masks_to_images(images, masks, patch_size):
    """
    Apply binary masks to images for visualization.

    Args:
        images (np.ndarray): [B, H, W, 3] images
        masks (torch.Tensor): [B, h, w] binary masks
        patch_size (int): Size of each patch

    Returns:
        masked_images (np.ndarray): [B, H, W, 3] masked images
    """
    B = images.shape[0]
    H, W = images.shape[1], images.shape[2]
    h, w = masks.shape[1], masks.shape[2]

    # Upsample masks to image size
    upsampled_masks = []
    for i in range(B):
        # Convert mask to numpy
        mask_np = masks[i].detach().cpu().numpy()

        # Upsample using nearest neighbor
        mask_upsampled = np.repeat(np.repeat(mask_np, patch_size, axis=0), patch_size, axis=1)

        # Make sure the upsampled mask matches the image size
        mask_upsampled = mask_upsampled[:H, :W]

        upsampled_masks.append(mask_upsampled)

    upsampled_masks = np.array(upsampled_masks)

    # Create masked images
    masked_images = []
    for i in range(B):
        # Create a copy of the image
        masked_img = images[i].copy()

        # Apply mask
        mask = upsampled_masks[i]

        # Create mask overlay: gray out masked regions
        mask_overlay = np.ones_like(masked_img) * 0.5  # mid-gray

        # Apply mask overlay to masked regions
        for c in range(3):
            masked_img[:, :, c] = np.where(mask == 0, mask_overlay[:, :, c], masked_img[:, :, c])

        masked_images.append(masked_img)

    return np.array(masked_images)


def visualize_continual_learning_performance(accuracy_matrix, task_names=None, output_path=None, bg_color="#fef0e7"):
    """
    Visualize continual learning performance across tasks.

    Args:
        accuracy_matrix (np.ndarray): Matrix where accuracy_matrix[i, j]
                                     is the accuracy on task j after learning task i
        task_names (list, optional): Names of tasks
        output_path (str, optional): Path to save visualization
        bg_color (str): Background color for visualization
    """
    num_tasks = accuracy_matrix.shape[0]

    if task_names is None:
        task_names = [f"Task {i+1}" for i in range(num_tasks)]

    # Create figure
    plt.figure(figsize=(10, 6), facecolor=bg_color)

    # Plot accuracy matrix as heatmap
    plt.imshow(accuracy_matrix, cmap='viridis', vmin=0, vmax=100)

    # Add colorbar
    plt.colorbar(label='Accuracy (%)')

    # Add grid
    plt.grid(False)

    # Add labels
    plt.xticks(range(num_tasks), task_names, rotation=45)
    plt.yticks(range(num_tasks), [f"After {name}" for name in task_names])

    plt.xlabel('Evaluated on')
    plt.ylabel('Trained on')
    plt.title('Accuracy Matrix for Continual Learning')

    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, facecolor=bg_color, bbox_inches='tight', dpi=300)

    plt.show()


def visualize_feature_space(features, labels, method='tsne', output_path=None, bg_color="#fef0e7"):
    """
    Visualize feature space using dimensionality reduction.

    Args:
        features (np.ndarray): [N, D] feature vectors
        labels (np.ndarray): [N] labels
        method (str): Dimensionality reduction method ('tsne' or 'pca')
        output_path (str, optional): Path to save visualization
        bg_color (str): Background color for visualization
    """
    # Import necessary libraries
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(features)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(features)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'tsne' or 'pca'.")

    # Create figure
    plt.figure(figsize=(10, 8), facecolor=bg_color)

    # Get unique labels
    unique_labels = np.unique(labels)

    # Create scatter plot with different colors for each class
    for label in unique_labels:
        mask = labels == label
        plt.scatter(
            reduced_features[mask, 0],
            reduced_features[mask, 1],
            label=f"Class {label}",
            alpha=0.7
        )

    plt.legend()
    plt.title(f'Feature Space Visualization using {method.upper()}')
    plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, facecolor=bg_color, bbox_inches='tight', dpi=300)

    plt.show()