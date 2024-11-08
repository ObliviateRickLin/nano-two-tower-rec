import numpy as np
from typing import List
import torch

def compute_metrics(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    true_labels: torch.Tensor,
    k_list: List[int] = [5, 10, 20]
) -> dict:
    """
    Compute recommendation metrics
    
    Args:
        user_embeddings: User embeddings (batch_size, embedding_dim)
        item_embeddings: Item embeddings (batch_size, embedding_dim)
        true_labels: Ground truth labels (batch_size,)
        k_list: List of k values for metrics@k
    
    Returns:
        Dictionary containing metrics
    """
    # Compute similarity scores
    sim_matrix = torch.matmul(
        user_embeddings,
        item_embeddings.transpose(0, 1)
    )
    
    metrics = {}
    for k in k_list:
        # Top-k predictions
        _, topk_indices = torch.topk(sim_matrix, k, dim=1)
        
        # Compute Recall@k
        recall = compute_recall_at_k(topk_indices, true_labels, k)
        metrics[f'recall@{k}'] = recall.item()
        
        # Compute NDCG@k
        ndcg = compute_ndcg_at_k(topk_indices, true_labels, k)
        metrics[f'ndcg@{k}'] = ndcg.item()
        
        # Compute MRR@k
        mrr = compute_mrr_at_k(topk_indices, true_labels, k)
        metrics[f'mrr@{k}'] = mrr.item()
    
    return metrics

def compute_recall_at_k(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Compute Recall@K
    
    Args:
        predictions: Predicted item indices (batch_size, k)
        labels: Ground truth labels (batch_size,)
        k: Number of top items to consider
    """
    # Convert labels to set of relevant items
    relevant_items = labels.unsqueeze(1).expand_as(predictions)
    
    # Check if relevant items are in top-k predictions
    hits = (predictions == relevant_items).any(dim=1).float()
    
    return hits.mean()

def compute_ndcg_at_k(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Compute NDCG@K
    
    Args:
        predictions: Predicted item indices (batch_size, k)
        labels: Ground truth labels (batch_size,)
        k: Number of top items to consider
    """
    # Create position weights
    position_weights = 1.0 / torch.log2(torch.arange(k, device=predictions.device) + 2.0)
    
    # Check if predictions match labels
    hits = (predictions == labels.unsqueeze(1)).float()
    
    # Compute DCG
    dcg = (hits * position_weights.unsqueeze(0)).sum(dim=1)
    
    # Compute ideal DCG (always 1.0 for binary relevance)
    idcg = position_weights[0]
    
    return (dcg / idcg).mean()

def compute_mrr_at_k(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Compute MRR@K (Mean Reciprocal Rank)
    
    Args:
        predictions: Predicted item indices (batch_size, k)
        labels: Ground truth labels (batch_size,)
        k: Number of top items to consider
    """
    # Find position of relevant items
    hits = (predictions == labels.unsqueeze(1))
    
    # Get reciprocal rank (1/position)
    ranks = torch.arange(1, k + 1, device=predictions.device).float()
    rr = (hits / ranks.unsqueeze(0)).sum(dim=1)
    
    return rr.mean()