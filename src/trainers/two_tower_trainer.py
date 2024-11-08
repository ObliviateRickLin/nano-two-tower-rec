import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils.losses import InfoNCELoss
from utils.metrics import compute_metrics

class TwoTowerTrainer(BaseTrainer):
    """Trainer class for two-tower model"""
    
    def __init__(
        self,
        model,
        optimizer,
        device,
        temperature: float = 0.07
    ):
        super().__init__(model, optimizer, device)
        self.criterion = InfoNCELoss(temperature)
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc='Training') as pbar:
            for batch in pbar:
                user_ids = batch['user_ids'].to(self.device)
                item_ids = batch['item_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                user_embeddings, item_embeddings = self.model(user_ids, item_ids)
                loss = self.criterion(user_embeddings, item_embeddings, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def validate(self, valid_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_metrics = {}
        
        with torch.no_grad():
            for batch in valid_loader:
                user_ids = batch['user_ids'].to(self.device)
                item_ids = batch['item_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                user_embeddings, item_embeddings = self.model(user_ids, item_ids)
                
                # Compute metrics
                metrics = compute_metrics(
                    user_embeddings,
                    item_embeddings,
                    labels
                )
                
                # Accumulate metrics
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
        
        # Average metrics
        for k in total_metrics:
            total_metrics[k] /= len(valid_loader)
            
        return total_metrics 