from typing import Tuple, Dict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class AliECDataset(Dataset):
    """Dataset class for AliEC recommendation data"""
    
    def __init__(self, data_path: str, mode: str = 'train'):
        """
        Args:
            data_path: Path to the processed data directory
            mode: One of 'train', 'valid', or 'test'
        """
        self.mode = mode
        self.data = self._load_data(data_path)
        self._process_data()
        
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess data"""
        # Load interaction data
        df = pd.read_csv(f'{data_path}/{self.mode}_interactions.csv')
        
        # Load user and item features
        user_features = pd.read_csv(f'{data_path}/user_features.csv')
        item_features = pd.read_csv(f'{data_path}/item_features.csv')
        
        # Merge features
        df = df.merge(user_features, on='user_id', how='left')
        df = df.merge(item_features, on='item_id', how='left')
        
        return df
        
    def _process_data(self):
        """Process data for model input"""
        # Encode user and item IDs if not already encoded
        if 'user_idx' not in self.data.columns:
            user_encoder = LabelEncoder()
            self.data['user_idx'] = user_encoder.fit_transform(self.data['user_id'])
        
        if 'item_idx' not in self.data.columns:
            item_encoder = LabelEncoder()
            self.data['item_idx'] = item_encoder.fit_transform(self.data['item_id'])
            
        # Create interaction matrix for negative sampling
        if self.mode == 'train':
            self.interaction_matrix = self._create_interaction_matrix()
            
    def _create_interaction_matrix(self) -> torch.Tensor:
        """Create sparse interaction matrix for negative sampling"""
        num_users = self.data['user_idx'].nunique()
        num_items = self.data['item_idx'].nunique()
        
        # Create sparse matrix of positive interactions
        interactions = torch.zeros((num_users, num_items), dtype=torch.bool)
        for _, row in self.data.iterrows():
            interactions[row['user_idx'], row['item_idx']] = 1
            
        return interactions
        
    def _get_negative_samples(self, user_idx: int, num_neg: int = 4) -> torch.Tensor:
        """Sample negative items for a user"""
        pos_items = self.interaction_matrix[user_idx].nonzero().squeeze()
        neg_items = torch.randint(
            0,
            self.interaction_matrix.size(1),
            (num_neg,)
        )
        
        # Resample if negative items overlap with positive items
        mask = torch.isin(neg_items, pos_items)
        while mask.any():
            neg_items[mask] = torch.randint(
                0,
                self.interaction_matrix.size(1),
                (mask.sum(),)
            )
            mask = torch.isin(neg_items, pos_items)
            
        return neg_items
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        user_idx = row['user_idx']
        item_idx = row['item_idx']
        
        if self.mode == 'train':
            # Get negative samples for training
            neg_items = self._get_negative_samples(user_idx)
            
            # Combine positive and negative items
            items = torch.cat([torch.tensor([item_idx]), neg_items])
            labels = torch.zeros(len(items))
            labels[0] = 1  # First item is positive
            
            return {
                'user_ids': torch.tensor(user_idx, dtype=torch.long),
                'item_ids': items,
                'labels': labels
            }
        else:
            return {
                'user_ids': torch.tensor(user_idx, dtype=torch.long),
                'item_ids': torch.tensor(item_idx, dtype=torch.long),
                'labels': torch.tensor(1, dtype=torch.float)
            }

def get_dataloader(
    data_path: str,
    batch_size: int,
    mode: str = 'train',
    num_workers: int = 4
) -> DataLoader:
    """Create data loader for training/validation/testing"""
    dataset = AliECDataset(data_path, mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True
    ) 