import torch
import torch.nn as nn
from src.models.towers.user_tower import UserTower
from src.models.towers.item_tower import ItemTower
from typing import Tuple, Dict

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
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 从batch中提取特征
        user_ids = batch['user_features'][:, 0].long()  # 第一列是user_idx
        user_features = batch['user_features'][:, 1:]   # 其余列是统计特征
        
        business_ids = batch['business_features'][:, 0].long()  # 第一列是business_idx
        business_features = batch['business_features'][:, 1:]   # 其余列是统计特征
        category_features = batch['category_features']
        
        # 分别通过两个塔
        user_embeddings = self.user_tower(user_ids, user_features)
        item_embeddings = self.item_tower(business_ids, business_features, category_features)
        
        return user_embeddings, item_embeddings 