from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader

class BaseTrainer(ABC):
    """Abstract base class for model trainers"""
    
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
    @abstractmethod
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch"""
        pass
        
    @abstractmethod
    def validate(self, valid_loader: DataLoader) -> float:
        """Validate the model"""
        pass 