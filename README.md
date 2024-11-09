# Two-Tower Recommendation System

A production-ready, PyTorch-based two-tower recommendation system optimized for Kaggle's free GPU resources. This project uses a subset of the [Yelp dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) and employs state-of-the-art techniques for efficient training and inference.

## Overview
This project implements a two-tower neural network architecture tailored for large-scale retrieval tasks. The model independently learns representations for users and items, designed to run efficiently on Kaggle's P100 GPU environment.

## Project Structure
```
src/
├── data/
│   ├── raw/                # Raw Yelp dataset
│   ├── processed/          # Preprocessed data files
│   └── data_loader.py      # Data loading and processing utilities
├── models/
│   ├── towers/
│   │   ├── user_tower.py    # User tower architecture
│   │   └── item_tower.py    # Item tower architecture
│   ├── layers/
│   │   ├── attention.py     # Attention mechanisms
│   │   └── pooling.py       # Pooling operations
│   └── two_tower.py         # Main two-tower model
├── trainers/
│   ├── base_trainer.py      # Base trainer class
│   └── two_tower_trainer.py # Two-tower model trainer
├── utils/
│   ├── metrics.py           # Evaluation metrics
│   ├── losses.py            # Loss functions
│   └── config.py            # Configuration utilities
└── notebooks/
    └── train_kaggle.ipynb   # Kaggle training notebook
```

## Features
- **Efficient Implementation**: Mixed precision training (FP16), gradient checkpointing, memory-efficient embeddings, and optimized data loading.
- **Flexible Model Architecture**: Multi-head self-attention for user behavior, feature interaction layers, configurable tower structures.
- **Production Ready**: Modular design, comprehensive logging, model checkpointing, and robust configuration management.

## Dataset
Using a subset of the Yelp dataset:
- Rich feature set including user and business attributes, review text, and ratings
- Includes user demographics and item (business) characteristics

## Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+
- Kaggle environment with P100 GPU (16GB)

## Installation
```bash
git clone https://github.com/username/two-tower-rec.git
cd two-tower-rec
pip install -r requirements.txt
```

## Quick Start

### Data Preparation
```bash
python src/data/data_loader.py --data_dir data/raw --output_dir data/processed
```

### Training on Kaggle
1. Upload the project to Kaggle.
2. Open `notebooks/train_kaggle.ipynb`.
3. Select GPU as the accelerator.
4. Run the notebook cells to train the model.

## Model Architecture Details

### User Tower
- **Input Features**:
  - User ID embedding, demographic features, historical behavior sequence, user context features.
- **Architecture**:
  - Feature embedding layers, multi-head self-attention, feature interaction layer, MLP layers.

### Item Tower
- **Input Features**:
  - Business ID embedding, category features, business attributes, business context features.
- **Architecture**:
  - Feature embedding layers, feature interaction layer, MLP layers.

### Training Strategy
- Batch size: 512
- Mixed precision training (FP16)
- Gradient checkpointing
- Early stopping
- Learning rate: 1e-3
- Loss: InfoNCE loss
- Optimizer: AdamW
- Negative sampling ratio: 1:4

## Configuration Example
```yaml
model:
  user_tower:
    embedding_dim: 64
    hidden_dims: [256, 128]
    num_heads: 4
    dropout: 0.1
  item_tower:
    embedding_dim: 64
    hidden_dims: [256, 128]
    dropout: 0.1
training:
  batch_size: 512
  learning_rate: 0.001
  num_epochs: 30
```

## Memory Optimization for Kaggle
- Mixed precision training
- Gradient checkpointing
- Efficient data loading
- Batch size optimization
- Memory-efficient embeddings

## Contributing
Contributions are welcome! Please submit a Pull Request if you'd like to help improve the project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this code, please cite it as follows:
```bibtex
@misc{two-tower-rec,
  author = {Your Name},
  title = {Two-Tower Recommendation System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/username/two-tower-rec}
}
```
