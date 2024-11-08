import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

# Default configuration
DEFAULT_CONFIG = {
    'model': {
        'num_users': 491647,
        'num_items': 240130,
        'user_tower': {
            'embedding_dim': 64,
            'hidden_dims': [256, 128],
            'num_heads': 4,
            'dropout': 0.1
        },
        'item_tower': {
            'embedding_dim': 64,
            'hidden_dims': [256, 128],
            'dropout': 0.1
        }
    },
    'training': {
        'batch_size': 512,
        'learning_rate': 0.001,
        'num_epochs': 30,
        'early_stopping_patience': 5,
        'temperature': 0.07
    },
    'data': {
        'num_workers': 4,
        'train_ratio': 0.8,
        'valid_ratio': 0.1
    }
} 