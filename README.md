# Towards Stable and Transferable Ultrasound Diagnosis Across Domains via Continual Masked Ultrasound Image Modeling
This repository provides the official implementation of CMUIM from the paper "Towards Stable and Transferable Ultrasound Diagnosis Across Domains via Continual Masked Ultrasound Image Modeling".


## Installation
```bash
# Clone the repository
git clone https://github.com/zhcz328/CMUIM.git
cd CMUIM

# Create a virtual environment (recommended)
conda create -n cmuim python=3.8
conda activate cmuim

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
- PIL
- tqdm

## Usage
### Training
Train the model on a single task:
```bash
python main.py --mode train --output_dir ./output/task0 --n_tasks 1 --data_path /path/to/OS-OGD
```

Continual learning on multiple tasks:
```bash
python main.py --mode train --output_dir ./output/continual --n_tasks 5 --data_path /path/to/OS-OGD --scenario PUL-IL
```

### Evaluation
Evaluate a trained model:
```bash
python main.py --mode eval --output_dir ./output/continual --n_tasks 5 --data_path /path/to/OS-OGD/test
```

### Visualization
Visualize the masks and model outputs:
```bash
python main.py --mode visualize --output_dir ./output/continual --n_tasks 5 --data_path /path/to/OS-OGD/test
```

## Configuration
The model and training behavior can be configured via command line arguments or by modifying the configuration file:
```bash
python main.py --embed_dim 768 --depth 12 --d_state 16 --mask_ratio 0.75 --lr 1e-4 --epochs 100 --batch_size 128
```

Key parameters:
- `embed_dim`: Embedding dimension (default: 768)
- `depth`: Number of transformer blocks (default: 12)
- `d_state`: State dimension for SSM (default: 16)
- `mask_ratio`: Fraction of patches to mask (default: 0.75)
- `lr`: Learning rate (default: 1e-3)
- `epochs`: Number of epochs per task (default: 100)
- `batch_size`: Batch size (default: 128)
- `buffer_size`: Memory buffer size per task (default: 800 for PUL-IL, 400 for others)

## Dataset Structure
The OS-OGD dataset is organized in the following format:
```
OS-OGD/
├── PUL-IL/
│   ├── task1_EP/
│   │   ├── train/
│   │   │   ├── class1/
│   │   │   ├── class2/
│   │   │   └── ...
│   │   └── val/
│   ├── task2_M2LP-1/
│   ├── task3_M2LP-2/
│   ├── task4_M2LP-3/
│   └── task5_GY/
├── IHUL-IL/
│   ├── task1_Shenzhen/
│   ├── task2_Chongqing/
│   ├── task3_Hunan/
│   ├── task4_Hainan/
│   └── task5_Guizhou/
└── DSUL-IL/
    ├── task1_Samsung/
    ├── task2_Philips/
    ├── task3_GE/
    ├── task4_Mindray/
    └── task5_Others/
```

## Continual Learning Scenarios
CMUIM supports three continual learning scenarios:

### 1. PUL-IL (Cross-period Ultrasound Incremental Learning)
- **Tasks**: Early Pregnancy (EP) → Mid-to-late Pregnancy (M2LP-1/2/3) → Gynecological (GY)
- **Classes**: 40 total (8 per task)
- **Usage**: `python main.py --scenario PUL-IL`

### 2. IHUL-IL (Interhospital Ultrasound Incremental Learning)
- **Tasks**: 5 different hospitals from various regions in China
- **Classes**: 24 (from 2nd/3rd trimester pregnancy)
- **Usage**: `python main.py --scenario IHUL-IL`

### 3. DSUL-IL (Device-specific Ultrasound Incremental Learning)
- **Tasks**: Different ultrasound device manufacturers
- **Classes**: 24 (Mid-to-late Pregnancy data)
- **Usage**: `python main.py --scenario DSUL-IL`


## Citation
If you use this code or the OS-OGD dataset in your research, please cite.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
This work was approved by the institutional ethics committee (approval number SFYLS[2022]068).
