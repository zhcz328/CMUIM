"""
Configuration settings for the S3UM architecture.

This module contains all configurable parameters for model architecture,
training, evaluation, and visualization.
"""

import os
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any


@dataclass
class ModelConfig:
    """
    Model architecture configuration.
    """
    # Embedding dimensions
    embed_dim: int = 768
    d_state: int = 16

    # Network structure
    depth: int = 3
    d_conv: int = 4
    dropout: float = 0.1

    # Layer specific configurations
    norm_layer: str = "LayerNorm"  # Options: "LayerNorm", "RMSNorm"

    # SSM kernel parameters
    dt_min: float = 0.001
    dt_max: float = 0.1


@dataclass
class DataConfig:
    """
    Data processing configuration.
    """
    # Image dimensions
    image_size: int = 224
    patch_size: int = 16

    # Data augmentation
    use_augmentation: bool = True
    random_crop: bool = True
    random_flip: bool = True
    color_jitter: float = 0.4

    # Normalization parameters
    mean: List[float] = (0.485, 0.456, 0.406)
    std: List[float] = (0.229, 0.224, 0.225)


@dataclass
class TrainingConfig:
    """
    Training configuration.
    """
    # Basic training parameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.05

    # Learning rate scheduler
    lr_scheduler: str = "cosine"  # Options: "cosine", "linear", "step"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Masking parameters
    mask_ratio: float = 0.75


@dataclass
class PathConfig:
    """
    Path configuration for datasets and model checkpoints.
    """
    # Base directories
    base_dir: str = "/archive/zhuchunzheng/tmi_CMUIM"
    data_dir: str = "/archive/zhuchunzheng/dafenlei/midlate_500_resize"

    # Output directories
    output_dir: str = os.path.join(base_dir, "output_dir")
    log_dir: str = os.path.join(output_dir, "logs")
    checkpoint_dir: str = os.path.join(output_dir, "checkpoints")
    visualization_dir: str = os.path.join(output_dir, "visualization_pdfs")

    # Pretrained model path
    pretrained_model_path: Optional[str] = os.path.join(base_dir, "output_dir_421/checkpoint-399.pth")


@dataclass
class VisualizationConfig:
    """
    Visualization configuration.
    """
    # Visualization settings
    save_visualizations: bool = True
    create_pdf: bool = True
    create_svg: bool = True

    # Plotting style
    background_color: str = "#fef0e7"
    figure_size: tuple = (6, 6)
    dpi: int = 300

    # Colormap settings
    colormap: str = "viridis"


@dataclass
class S3UMConfig:
    """
    Complete configuration for the S3UM project.
    """
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    paths: PathConfig = PathConfig()
    visualization: VisualizationConfig = VisualizationConfig()

    # System configuration
    seed: int = 42
    device: str = "cuda"  # Options: "cuda", "cpu"
    num_workers: int = 8
    pin_memory: bool = True

    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with values from a dictionary.

        Args:
            config_dict: Dictionary with configuration values
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'S3UMConfig':
        """
        Create configuration from a dictionary.

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            S3UMConfig instance
        """
        config = cls()
        config.update(config_dict)
        return config


# Default configuration
DEFAULT_CONFIG = S3UMConfig()


def get_config(config_path: Optional[str] = None) -> S3UMConfig:
    """
    Get configuration from file or use default.

    Args:
        config_path: Path to configuration file (optional)

    Returns:
        S3UMConfig instance
    """
    if config_path is None:
        return DEFAULT_CONFIG

    # Load configuration from file (YAML, JSON, etc.)
    # Implementation depends on the chosen file format
    raise NotImplementedError("Loading configuration from file is not implemented yet")