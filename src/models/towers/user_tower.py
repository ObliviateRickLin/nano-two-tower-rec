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
        
        # Build MLP layers
        layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        x = self.user_embedding(user_ids)
        return self.mlp(x) 