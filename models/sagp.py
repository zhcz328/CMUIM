"""
Semantic-Aware Gradient Perturbation (SAGP) module for S3UM.

This module implements a gradient perturbation strategy that operates
in the semantic embedding space to enhance feature separability while
maintaining semantic consistency across tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAwareGradientPerturbation:
    """
    Semantic-Aware Gradient Perturbation for buffer samples in continual learning.

    This technique enhances representation quality by creating meaningful perturbations
    that operate in the semantic subspace spanned by task representations, improving
    cross-task knowledge transfer and feature separability.
    """

    def __init__(self, eta=0.01):
        """
        Initialize SAGP.

        Args:
            eta (float): Perturbation magnitude
        """
        self.eta = eta

    def generate_perturbations(self, model_current, model_previous, samples, device):
        """
        Generate semantically-aware perturbations for buffer samples.

        Args:
            model_current (nn.Module): Current task model
            model_previous (nn.Module): Previous task model
            samples (torch.Tensor): Buffer samples to perturb
            device (torch.device): Device to perform computation on

        Returns:
            perturbed_samples (torch.Tensor): Samples with semantic perturbations
        """
        # Ensure models are in evaluation mode during perturbation generation
        model_current.eval()
        model_previous.eval()

        # Create a copy of samples that requires gradients
        samples_clone = samples.clone().detach().to(device).requires_grad_(True)

        # Get task-specific gradients through dual pathways
        with torch.enable_grad():
            # Current task gradient
            loss_current = self._compute_reconstruction_loss(model_current, samples_clone)
            grad_current = torch.autograd.grad(
                loss_current, samples_clone, create_graph=False, retain_graph=True
            )[0]

            # Previous task gradient
            loss_previous = self._compute_reconstruction_loss(model_previous, samples_clone)
            grad_previous = torch.autograd.grad(
                loss_previous, samples_clone, create_graph=False, retain_graph=True
            )[0]

            # Gradient difference between tasks
            grad_diff = grad_current - grad_previous

            # Extract feature representations for semantic subspace projection
            with torch.no_grad():
                z_current = self._extract_features(model_current, samples_clone)
                z_previous = self._extract_features(model_previous, samples_clone)

            # Project gradient difference onto semantic subspace
            semantic_grad = self._project_to_semantic_subspace(
                grad_diff, z_current, z_previous
            )

            # Generate perturbed samples
            perturbed_samples = self._apply_perturbation(samples_clone, semantic_grad)

        # Restore model training state
        if model_current.training:
            model_current.train()
        if model_previous.training:
            model_previous.train()

        return perturbed_samples

    def _compute_reconstruction_loss(self, model, samples):
        """
        Compute reconstruction loss for a model.

        Args:
            model (nn.Module): Model to compute loss for
            samples (torch.Tensor): Input samples

        Returns:
            loss (torch.Tensor): Reconstruction loss
        """
        if hasattr(model, 'forward_features'):
            # For models with separate feature extraction
            features = model.forward_features(samples)
            loss = F.mse_loss(features, torch.zeros_like(features))
        else:
            # For standard autoencoder models
            reconstructions = model(samples)
            loss = F.mse_loss(reconstructions, samples)

        return loss

    def _extract_features(self, model, samples):
        """
        Extract feature representations from a model.

        Args:
            model (nn.Module): Model to extract features from
            samples (torch.Tensor): Input samples

        Returns:
            features (torch.Tensor): Extracted features
        """
        if hasattr(model, 'encode') and callable(model.encode):
            # For models with explicit encode method
            return model.encode(samples)
        elif hasattr(model, 'forward_features') and callable(model.forward_features):
            # For models with forward_features method
            return model.forward_features(samples)
        else:
            # For general models, use intermediate layer
            # This is a simplification - actual implementation would need
            # to access appropriate intermediate representations
            x = samples
            if hasattr(model, 'patch_embed'):
                x = model.patch_embed(x)
            if hasattr(model, 'blocks') and len(model.blocks) > 0:
                x = model.blocks[0](x)
            return x.mean(dim=1)  # Global pooling

    def _project_to_semantic_subspace(self, gradient, z_current, z_previous):
        """
        Project gradient onto semantic subspace spanned by feature vectors.

        Args:
            gradient (torch.Tensor): Gradient to project
            z_current (torch.Tensor): Current task feature representation
            z_previous (torch.Tensor): Previous task feature representation

        Returns:
            projected_grad (torch.Tensor): Projected gradient
        """
        # Flatten spatial dimensions if needed
        if len(gradient.shape) > 2:
            b, *spatial_dims = gradient.shape
            gradient_flat = gradient.reshape(b, -1)
            z_dim = z_current.shape[-1]

            # Initialize projected gradient tensor
            projected_grad_flat = torch.zeros_like(gradient_flat)

            # Process each sample in batch
            for i in range(b):
                # Create basis for semantic subspace
                z_basis = torch.stack([
                    z_current[i], z_previous[i]
                ], dim=1)  # [D, 2]

                # Orthogonalize basis (optional but more stable)
                q, r = torch.linalg.qr(z_basis)
                basis = q  # Orthonormal basis

                # Project gradient onto semantic subspace
                projection = torch.mm(
                    torch.mm(gradient_flat[i:i + 1], basis),
                    basis.t()
                )

                projected_grad_flat[i:i + 1] = projection

            # Reshape back to original dimensions
            projected_grad = projected_grad_flat.reshape(gradient.shape)

        else:
            # Direct batch projection for already flattened gradients
            b = gradient.shape[0]
            projected_grad = torch.zeros_like(gradient)

            for i in range(b):
                z_basis = torch.stack([
                    z_current[i], z_previous[i]
                ], dim=1)

                q, r = torch.linalg.qr(z_basis)
                basis = q

                projection = torch.mm(
                    torch.mm(gradient[i:i + 1], basis),
                    basis.t()
                )

                projected_grad[i:i + 1] = projection

        return projected_grad

    def _apply_perturbation(self, samples, semantic_grad):
        """
        Apply normalized perturbation to samples.

        Args:
            samples (torch.Tensor): Original samples
            semantic_grad (torch.Tensor): Projected semantic gradient

        Returns:
            perturbed_samples (torch.Tensor): Perturbed samples
        """
        # Normalize gradient for stable perturbation magnitude
        grad_norm = torch.norm(semantic_grad, dim=tuple(range(1, len(semantic_grad.shape))), keepdim=True)
        normalized_grad = semantic_grad / (grad_norm + 1e-8)

        # Apply perturbation
        perturbed_samples = samples + self.eta * normalized_grad

        # Ensure values remain within valid range [0, 1]
        perturbed_samples = torch.clamp(perturbed_samples, 0.0, 1.0)

        return perturbed_samples