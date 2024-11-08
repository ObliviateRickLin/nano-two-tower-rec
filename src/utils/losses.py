import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            user_embeddings: (batch_size, embedding_dim)
            item_embeddings: (batch_size, embedding_dim)
            labels: (batch_size,) binary labels indicating positive pairs
        """
        # Compute similarity matrix
        sim_matrix = torch.matmul(
            user_embeddings, 
            item_embeddings.transpose(0, 1)
        ) / self.temperature
        
        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = -torch.sum(labels * log_prob) / labels.sum()
        
        return loss 