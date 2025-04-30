"""
Configuration parameters for CMUIM.

This module provides a comprehensive configuration for model architecture,
training, optimization, and dataset parameters.
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


def get_args_parser():
    """
    Create and return the argument parser for CMUIM.

    Returns:
        argparse.ArgumentParser: Argument parser with CMUIM configuration parameters.
    """
    parser = argparse.ArgumentParser('CMUIM', add_help=False)

    # Model architecture parameters
    parser.add_argument('--model', default='masking_net', type=str,
                        help='Model architecture to use')
    parser.add_argument('--embed_dim', default=768, type=int,
                        help='Embedding dimension for vision transformer')
    parser.add_argument('--depth', default=5, type=int,
                        help='Number of S6 blocks')
    parser.add_argument('--d_state', default=16, type=int,
                        help='State dimension for SSM')
    parser.add_argument('--d_conv', default=4, type=int,
                        help='Kernel size for depth-wise convolution in S6 block')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout rate for the model')
    parser.add_argument('--dt_min', default=0.001, type=float,
                        help='Minimum discretization step size for SSM')
    parser.add_argument('--dt_max', default=0.1, type=float,
                        help='Maximum discretization step size for SSM')

    # Masking parameters
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Fraction of patches to mask')
    parser.add_argument('--lambda_area', default=1.0, type=float,
                        help='Weight for area regularization')
    parser.add_argument('--lambda_div', default=0.2, type=float,
                        help='Weight for diversity regularization')

    # Data parameters
    parser.add_argument('--data_path', default='/path/to/dataset', type=str,
                        help='Dataset path')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Images input size')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers for data loading')

    # Optimization parameters
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--weight_decay', default=0.05, type=float,
                        help='Weight decay')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of training epochs')
    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help='Number of epochs for learning rate warmup')
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='Minimum learning rate')

    # Continual learning parameters
    parser.add_argument('--n_tasks', default=5, type=int,
                        help='Number of tasks for continual learning')
    parser.add_argument('--buffer_size', default=500, type=int,
                        help='Size of memory buffer per task for continual learning')
    parser.add_argument('--alpha', default=0.8, type=float,
                        help='Weight for balancing task-specific and cross-task losses')
    parser.add_argument('--beta', default=0.2, type=float,
                        help='Weight for semantic-aware gradient perturbation')
    parser.add_argument('--temperature', default=0.05, type=float,
                        help='Temperature for contrastive learning')

    # System parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use for training/testing')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', default='./output',
                        help='Path to save outputs')
    parser.add_argument('--log_dir', default='./logs',
                        help='Path to save logs')
    parser.add_argument('--save_freq', default=10, type=int,
                        help='Frequency of saving checkpoints')

    # Distributed training parameters
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='Distributed backend')
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='Node rank for distributed training')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Local rank for distributed training')
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training')

    # Visualization parameters
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization')
    parser.add_argument('--bg_color', default='#fef0e7', type=str,
                        help='Background color for visualization')
    parser.add_argument('--fig_size', default=6, type=int,
                        help='Figure size for visualization')
    parser.add_argument('--dpi', default=300, type=int,
                        help='DPI for visualization')

    return parser


@dataclass
class CMUIMConfig:
    """Configuration parameters for CMUIM model and training."""

    # Model architecture
    embed_dim: int = 768
    depth: int = 5
    d_state: int = 16
    d_conv: int = 4
    dropout: float = 0.1
    dt_min: float = 0.001
    dt_max: float = 0.1

    # Masking parameters
    mask_ratio: float = 0.75
    lambda_area: float = 1.0
    lambda_div: float = 0.2

    # Data parameters
    data_path: str = '/path/to/dataset'
    input_size: int = 224
    batch_size: int = 64
    num_workers: int = 8

    # Optimization parameters
    lr: float = 1e-4
    weight_decay: float = 0.05
    epochs: int = 100
    warmup_epochs: int = 10
    min_lr: float = 1e-6

    # Continual learning parameters
    n_tasks: int = 5
    buffer_size: int = 500
    alpha: float = 0.8
    beta: float = 0.2
    temperature: float = 0.05

    # System parameters
    device: str = 'cuda'
    seed: int = 42
    output_dir: str = './output'
    log_dir: str = './logs'
    save_freq: int = 10

    # Visualization parameters
    visualize: bool = True
    bg_color: str = '#fef0e7'
    fig_size: int = 6
    dpi: int = 300

    @classmethod
    def from_args(cls, args):
        """Create config from parsed arguments."""
        config = cls()
        for key, value in vars(args).items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config