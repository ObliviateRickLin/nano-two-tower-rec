import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_raw_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw AliEC dataset"""
    data_dir = Path(data_dir)
    
    # Load interactions
    interactions = pd.read_csv(data_dir / 'user_behaviors.csv')
    
    # Load user features
    user_features = pd.read_csv(data_dir / 'user_profiles.csv')
    
    # Load item features
    item_features = pd.read_csv(data_dir / 'item_info.csv')
    
    return interactions, user_features, item_features

def process_features(
    interactions: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process user and item features"""
    
    # Process user features
    user_features['age_bucket'] = pd.qcut(user_features['age'], q=10, labels=False)
    user_features = pd.get_dummies(user_features, columns=['gender', 'age_bucket'])
    
    # Process item features
    item_features['price_bucket'] = pd.qcut(item_features['price'], q=10, labels=False)
    item_features = pd.get_dummies(
        item_features,
        columns=['category_id', 'price_bucket']
    )
    
    return interactions, user_features, item_features

def split_data(
    interactions: pd.DataFrame,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    time_based: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/valid/test sets"""
    
    if time_based:
        # Sort by timestamp
        interactions = interactions.sort_values('timestamp')
        
        # Calculate split points
        n = len(interactions)
        train_idx = int(n * train_ratio)
        valid_idx = int(n * (train_ratio + valid_ratio))
        
        # Split data
        train = interactions.iloc[:train_idx]
        valid = interactions.iloc[train_idx:valid_idx]
        test = interactions.iloc[valid_idx:]
    else:
        # Random split
        train, temp = train_test_split(
            interactions,
            train_size=train_ratio,
            random_state=42
        )
        valid, test = train_test_split(
            temp,
            train_size=valid_ratio/(1-train_ratio),
            random_state=42
        )
    
    return train, valid, test

def main():
    """Main preprocessing function"""
    # Load raw data
    interactions, user_features, item_features = load_raw_data('data/raw')
    
    # Process features
    interactions, user_features, item_features = process_features(
        interactions,
        user_features,
        item_features
    )
    
    # Split data
    train, valid, test = split_data(interactions)
    
    # Create output directory
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    train.to_csv(output_dir / 'train_interactions.csv', index=False)
    valid.to_csv(output_dir / 'valid_interactions.csv', index=False)
    test.to_csv(output_dir / 'test_interactions.csv', index=False)
    user_features.to_csv(output_dir / 'user_features.csv', index=False)
    item_features.to_csv(output_dir / 'item_features.csv', index=False)

if __name__ == '__main__':
    main() 