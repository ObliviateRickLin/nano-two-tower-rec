import torch
import torch.nn as nn

class UserTower(nn.Module):
    """User tower of the two-tower model"""
    
    def __init__(
        self,
        num_users: int,
        embedding_dim: int,
        hidden_dims: list,
        num_heads: int,
        dropout: float
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        # 用户统计特征的维度：review_count + average_stars + yelping_days = 3
        self.user_features_dim = 3
        
        # 修改MLP的输入维度，加入统计特征
        layers = []
        input_dim = embedding_dim + self.user_features_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, user_ids: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        # user_ids: [batch_size]
        # user_features: [batch_size, user_features_dim]
        
        id_embedding = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        x = torch.cat([id_embedding, user_features], dim=1)  # [batch_size, embedding_dim + user_features_dim]
        return self.mlp(x) 