import torch
import torch.nn as nn

class ItemTower(nn.Module):
    """Item tower of the two-tower model"""
    
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        hidden_dims: list,
        dropout: float
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
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
        
    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        x = self.item_embedding(item_ids)
        return self.mlp(x) 