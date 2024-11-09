import torch
import torch.nn as nn
from src.models.towers.user_tower import UserTower
from src.models.towers.item_tower import ItemTower
from typing import Tuple

class TwoTowerModel(nn.Module):
    """Two-tower recommendation model"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.user_tower = UserTower(
            num_users=config['num_users'],
            embedding_dim=config['user_tower']['embedding_dim'],
            hidden_dims=config['user_tower']['hidden_dims'],
            num_heads=config['user_tower']['num_heads'],
            dropout=config['user_tower']['dropout']
        )
        
        self.item_tower = ItemTower(
            num_items=config['num_items'],
            embedding_dim=config['item_tower']['embedding_dim'],
            hidden_dims=config['item_tower']['hidden_dims'],
            dropout=config['item_tower']['dropout']
        )
        
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_embeddings = self.user_tower(user_ids)
        item_embeddings = self.item_tower(item_ids)
        return user_embeddings, item_embeddings 