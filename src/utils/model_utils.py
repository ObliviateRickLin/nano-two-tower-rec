import torch
from typing import Dict, Any
import json
from pathlib import Path

def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_dir: str,
    model_name: str = 'two_tower'
):
    """Save model checkpoint"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model state
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(
        checkpoint,
        save_dir / f'{model_name}_epoch_{epoch}.pt'
    )
    
    # Save metrics history
    metrics_file = save_dir / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics_history = json.load(f)
    else:
        metrics_history = []
    
    metrics_history.append({
        'epoch': epoch,
        **metrics
    })
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_history, f, indent=2)

def load_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str
) -> tuple:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint['epoch'], checkpoint['metrics'] 