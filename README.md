# CatDiff: Diffusion Model for Catalyst Structure Generation

CatDiff is a diffusion-based generative model for catalyst structure generation. The model can simultaneously generate composition, lattice parameters, and atomic coordinates of crystal structures.

## Key Features

- **Composition Prediction**: Predicts appropriate number of atoms and types based on given element combinations and ratios
- **Structure Generation**: Simultaneously generates lattice parameters and atomic coordinates
- **Adsorbate Handling**: Supports adding adsorbates to surface structures
- **Flexible Coordinate System**: Supports both fractional and Cartesian coordinates

## Model Architecture

### CSPNet (Crystal Structure Prediction Network)
- Graph neural network for encoding crystal structures
- Includes atom embeddings, lattice information processing, and adsorbate handling
- Edge generation methods: fully connected (fc) or k-nearest neighbors (knn)

### Diffusion Model
- Generates crystal structures progressively from noise
- Uses separate loss functions for composition, lattice, and coordinates
- Implements PC sampling strategy

## Configuration

The project uses Hydra for configuration management. Key configuration files:
- `conf/model/diffusion.yaml`: Diffusion model settings
- `conf/model/decoder/cspnet.yaml`: CSPNet settings
- `conf/data/*.yaml`: Dataset configurations
- `conf/optim/default.yaml`: Optimization settings

## Datasets

Supported datasets:
- **OC20**: Open Catalyst 2020 dataset
  - Contains 130M+ relaxation trajectories
  - Focuses on catalyst surface-adsorbate interactions
- **OC20-Dense**: Dense subset of OC20
  - Carefully curated for high-quality training
  - Contains more complete relaxation trajectories

## Usage

### Environment Setup
```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate catdiff
```

### Training
```python
# Example training configuration
python diffcsp/run.py \
    model=diffusion \
    data=oc20_2M \
    optim=default
```

### Structure Generation
```python
# Run generation script
python scripts/generation.py \
    checkpoint=path/to/checkpoint \
    num_samples=100
```

## Project Structure
```
diffcsp/
├── pl_modules/
│   ├── diffusion.py     # Diffusion model implementation
│   ├── cspnet.py        # Crystal structure encoder
│   └── model.py         # Base model class
├── pl_data/
│   ├── datamodule.py    # PyTorch Lightning data module
│   └── dataset.py       # Dataset implementation
└── common/
    ├── data_utils.py    # Data processing utilities
    └── utils.py         # General utility functions
```

## Notes

- Model is implemented using PyTorch Lightning
- Git LFS is recommended for large dataset handling
- Training data should be stored in `data/` directory (git-ignored)
- The model is primarily designed and tested on the OC20 dataset for catalyst structure generation