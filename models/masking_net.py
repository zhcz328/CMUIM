"""
MaskingNet model for the S3UM architecture.
This module implements a Mamba-Enhanced Anatomical Structure Selection network.
"""

import torch
import torch.nn as nn
import numpy as np
from util.positional_embedding import get_2d_sincos_pos_embed
from S3UM.s3um_block import S3UMBlock


class MaskingNet(nn.Module):
    """
    Masking network with Mamba-Enhanced Anatomical Structure Selection (S3UM).

    This model leverages state-space models to identify and select anatomical
    structures in medical images through an innovative masking approach.
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
        """
        Initialize the MaskingNet model.

        Args:
            num_tokens: Number of image patches/tokens
            embed_dim: Embedding dimension
            depth: Number of S6 blocks
            d_state: State dimension for SSM
            dropout: Dropout rate
            norm_layer: Normalization layer
        """
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
        """
        Initialize the weights of the model.

        Args:
            m: Module to initialize
        """
        if isinstance(m, nn.Linear):
            # Xavier initialization for linear layers
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_weights(self):
        """
        Initialize the weights of the model with proper strategies
        for different components.
        """
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
        Forward pass of the MaskingNet model.

        Args:
            x: [B, N, D] tensor of patch tokens

        Returns:
            mask_probs: [B, N] mask probabilities
        """
        # Add class token and positional embedding
        B, N, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # Apply S6 blocks
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
        Generate binary mask by selecting top-k probabilities.

        Args:
            x: [B, N, D] tensor of patch tokens
            mask_ratio: Fraction of patches to mask

        Returns:
            mask: [B, N] binary mask (1=keep, 0=mask)
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