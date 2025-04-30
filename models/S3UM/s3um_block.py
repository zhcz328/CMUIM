"""
S3UM Block implementation for the S3UM architecture.
This module provides a Mamba-style architecture featuring selective SSM mechanisms.
"""

import torch
import torch.nn as nn
from .ssm_kernel import SSMKernel


class S3UMBlock(nn.Module):
    """
    S3UMBlock with Mamba-style architecture featuring selective SSM.

    This advanced block combines:
    - Selective State-Space Models for long-range dependencies
    - Depth-wise convolutions for local feature extraction
    - Gating mechanisms for adaptive information flow
    - Residual connections for stable optimization
    """

    def __init__(self, dim, d_state=16, d_conv=4, dropout=0.1):
        """
        Initialize the S3UM Block.

        Args:
            dim: Hidden dimension
            d_state: State dimension for SSM
            d_conv: Kernel size for depth-wise convolution
            dropout: Dropout rate
        """
        super().__init__()
        # Layer normalization for stable training
        self.norm = nn.LayerNorm(dim)

        # Left branch - SSM pathway
        self.in_proj = nn.Linear(dim, dim)
        # Depth-wise convolution for local feature extraction
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

    def forward(self, x):
        """
        Forward pass of the S3UM Block.

        Args:
            x: [B, L, D] input tensor

        Returns:
            output: [B, L, D] output tensor
        """
        # Save residual connection
        residual = x

        # Layer normalization
        x_norm = self.norm(x)

        # Left branch processing
        left = self.in_proj(x_norm)

        # Convolution processing - adjust dimensions [B, L, D] -> [B, D, L]
        left = left.transpose(1, 2)
        left = self.conv1d(left)[:, :, :-self.conv1d.padding[0]]  # Remove padding-induced extra part
        left = left.transpose(1, 2)  # Restore [B, L, D]

        # Activation function
        left = self.activation(left)

        # Right branch - compute gating values
        gate = self.activation(x_norm)

        # SSM processing
        left = self.ssm(left)

        # Gating mechanism - multiply left branch and right branch
        y = left * gate

        # Project and apply dropout
        y = self.out_proj(y)
        y = self.dropout(y)

        # Residual connection
        output = residual + y

        return output