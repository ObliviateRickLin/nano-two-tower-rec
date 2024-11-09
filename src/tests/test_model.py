import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
import pytest
from src.models.two_tower import TwoTowerModel
from src.utils.config import DEFAULT_CONFIG

class TestTwoTowerModel:
    """Test shape consistency of two-tower model"""
    
    @pytest.fixture(scope="class")
    def model(self):
        """Create model instance for testing"""
        return TwoTowerModel(DEFAULT_CONFIG['model'])
    
    @pytest.fixture(scope="class")
    def batch_inputs(self):
        """Create dummy batch inputs"""
        batch_size = 32
        user_ids = torch.randint(0, DEFAULT_CONFIG['model']['num_users'], (batch_size,))
        item_ids = torch.randint(0, DEFAULT_CONFIG['model']['num_items'], (batch_size,))
        return {
            'batch_size': batch_size,
            'user_ids': user_ids,
            'item_ids': item_ids
        }
    
    def test_user_tower_output_shape(self, model, batch_inputs):
        """Test if user tower outputs correct embedding shape"""
        user_emb, _ = model(batch_inputs['user_ids'], batch_inputs['item_ids'])
        expected_shape = (
            batch_inputs['batch_size'], 
            DEFAULT_CONFIG['model']['user_tower']['hidden_dims'][-1]
        )
        assert user_emb.shape == expected_shape, \
            f"User embedding shape {user_emb.shape} != expected shape {expected_shape}"
    
    def test_item_tower_output_shape(self, model, batch_inputs):
        """Test if item tower outputs correct embedding shape"""
        _, item_emb = model(batch_inputs['user_ids'], batch_inputs['item_ids'])
        expected_shape = (
            batch_inputs['batch_size'], 
            DEFAULT_CONFIG['model']['item_tower']['hidden_dims'][-1]
        )
        assert item_emb.shape == expected_shape, \
            f"Item embedding shape {item_emb.shape} != expected shape {expected_shape}"
    
    def test_embedding_dimension_match(self, model, batch_inputs):
        """Test if user and item embeddings have matching dimensions"""
        user_emb, item_emb = model(batch_inputs['user_ids'], batch_inputs['item_ids'])
        assert user_emb.shape[1] == item_emb.shape[1], \
            f"User embedding dim {user_emb.shape[1]} != Item embedding dim {item_emb.shape[1]}"