"""
S3UM: Selective State Space Models for Ultrasound Masking

A modular implementation of Mamba-Enhanced Anatomical Structure Selection.
"""

from .models.masking_net import MaskingNet
from .positional_embedding import (
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_from_grid,
    get_1d_sincos_pos_embed_from_grid
)

__version__ = '0.1.0'
__all__ = ['MaskingNet']

__all__ = [
    'get_2d_sincos_pos_embed',
    'get_2d_sincos_pos_embed_from_grid',
    'get_1d_sincos_pos_embed_from_grid'
]