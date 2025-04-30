# CMUIM: A New Benchmark For Continuous Self-supervised Learning In Ultrasound Imaging

This repository implements the CMUIM as described in the paper "CMUIM: A New Benchmark For Continuous Self-supervised Learning In Ultrasound Imaging".

## Overview

CMUIM is designed to address the unique challenges of ultrasound image analysis in a continual learning setting. The framework includes:

1. **Selective State Space Ultrasound Masking (S3UM)**: A novel masking strategy that leverages state space models to identify anatomically significant regions in ultrasound images.

2. **Bi-level Optimization**: A two-stage optimization process that enhances both task-specific features and cross-task knowledge transfer.

3. **Semantic-Aware Gradient Perturbation (SAGP)**: A technique that improves feature separability and reduces catastrophic forgetting during task transitions.

## Architecture

The S3UM architecture consists of several key components:

- **SSMKernel**: Implements the selective state space model with input-dependent parameters for adaptive processing of ultrasound content.
- **S3UMBlock**: Combines selective state space models with convolutional and gating mechanisms for efficient feature extraction.
- **MaskingNet**: Creates masks that prioritize anatomically significant regions for reconstruction.
- **CMUIM**: The complete model that integrates masked autoencoding with continual learning strategies.

## Installation

```bash
# Clone the repository
git clone https://github.com/zhcz328/CMUIM.git
cd CMUIM

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- einops
- matplotlib
- numpy
- scikit-learn (for visualization tools)

## Usage

### Training

Train the model on a single task:

```bash
python main.py --mode train --output_dir ./output/task0 --n_tasks 1 --data_path /path/to/dataset
```

Continual learning on multiple tasks:

```bash
python main.py --mode train --output_dir ./output/continual --n_tasks 5 --data_path /path/to/dataset
```

### Evaluation

Evaluate a trained model:

```bash
python main.py --mode eval --output_dir ./output/continual --n_tasks 5 --data_path /path/to/testset
```

### Visualization

Visualize the masks and model outputs:

```bash
python main.py --mode visualize --output_dir ./output/continual --n_tasks 5 --data_path /path/to/testset
```

## Configuration

The model and training behavior can be configured via command line arguments or by modifying the configuration file:

```bash
python main.py --embed_dim 768 --depth 12 --d_state 16 --mask_ratio 0.75 --lr 1e-4 --epochs 100
```

Key parameters:
- `embed_dim`: Embedding dimension (default: 768)
- `depth`: Number of transformer blocks (default: 12)
- `d_state`: State dimension for SSM (default: 16)
- `mask_ratio`: Fraction of patches to mask (default: 0.75)
- `alpha`: Weight for balancing task-specific and cross-task losses (default: 0.8)
- `beta`: Weight for semantic-aware gradient perturbation (default: 0.2)

## Dataset Structure

The dataset should be organized in the following format:

```
dataset_root/
    ├── task1/
    │   ├── train/
    │   │   ├── class1/
    │   │   ├── class2/
    │   │   └── ...
    │   └── val/
    │       ├── class1/
    │       ├── class2/
    │       └── ...
    ├── task2/
    │   ├── train/
    │   ├── val/
    │   └── ...
    └── ...
```

## Continual Learning Scenarios

S3UM supports three continual learning scenarios:

1. **PUL-IL**: Cross-period ultrasound incremental learning
2. **IHUL-IL**: Interhospital ultrasound incremental learning
3. **DSUL-IL**: Device-specific ultrasound incremental learning