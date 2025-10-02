"""
SSM Kernel module for the S3UM architecture.
Implements the Selective State Space Model kernel with input-dependent parameters.
"""

import torch
import torch.nn as nn
import math
from einops import rearrange


class SSMKernel(nn.Module):
    """
    Selective State Space Model kernel for sequence modeling
    with input-dependent parameters (A, B, C, delta).

    This advanced implementation features:
      - Dynamic parameter generation based on input content
      - Selective mechanism for content-dependent processing
      - Stability constraints for robust convergence
    """

    def __init__(self, d_model, d_state, dt_min=0.001, dt_max=0.1):
        """
        Initialize the SSM kernel.

        Args:
            d_model: Embedding dimension
            d_state: State dimension
            dt_min: Minimum discretization step size
            dt_max: Maximum discretization step size
        """
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
        Forward pass of the SSM kernel.

        Args:
            x: [B, L, D] input tensor

        Returns:
            y: [B, L, D] output tensor
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


def selective_ssm_scan_with_dynamic_A_B_C(u, A_log, B_values, C_values, delta, dt, D=None):
    """
    Selective state space model scan operation with input-dependent A, B, and C

    Args:
        u: [B, D, L] input tensor
        A_log: [B, L, N] - input-dependent A matrix (in log space)
        B_values: [B, L, N] - input-dependent B matrix
        C_values: [B, L, N] - input-dependent C matrix
        delta: [B, D, L] - selective parameter
        dt: [D] - discretization step size
        D: [D] - optional direct term

    Returns:
        y: [B, D, L] output tensor
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
        # delta * h(t-1) + (1 - delta) * x
        x = delta[:, :, i:i + 1] * x + (1 - delta[:, :, i:i + 1]) * current_input

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