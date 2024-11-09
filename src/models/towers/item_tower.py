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
        
        # 商家统计特征维度：review_count + stars = 2
        self.business_features_dim = 2
        # 类别特征维度：10个类别的one-hot编码
        self.category_features_dim = 10
        
        # 修改MLP的输入维度
        layers = []
        input_dim = embedding_dim + self.business_features_dim + self.category_features_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        
    def forward(
        self, 
        item_ids: torch.Tensor, 
        business_features: torch.Tensor,
        category_features: torch.Tensor
    ) -> torch.Tensor:
        # item_ids: [batch_size]
        # business_features: [batch_size, business_features_dim]
        # category_features: [batch_size, category_features_dim]
        
        id_embedding = self.item_embedding(item_ids)  # [batch_size, embedding_dim]
        x = torch.cat([
            id_embedding, 
            business_features, 
            category_features
        ], dim=1)  # [batch_size, embedding_dim + business_features_dim + category_features_dim]
        return self.mlp(x) 