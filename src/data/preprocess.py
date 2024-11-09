import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_raw_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载Yelp原始数据集"""
    data_dir = Path(data_dir)
    
    # 加载评论数据（作为交互数据）
    reviews = pd.read_json(data_dir / 'yelp_academic_dataset_review.json', lines=True)
    reviews = reviews[['user_id', 'business_id', 'stars', 'date']]
    
    # 加载用户数据
    users = pd.read_json(data_dir / 'yelp_academic_dataset_user.json', lines=True)
    users = users[['user_id', 'review_count', 'yelping_since', 'average_stars']]
    
    # 加载商家数据
    businesses = pd.read_json(data_dir / 'yelp_academic_dataset_business.json', lines=True)
    businesses = businesses[['business_id', 'stars', 'review_count', 'categories']]
    
    return reviews, users, businesses

def process_features(
    reviews: pd.DataFrame,
    users: pd.DataFrame,
    businesses: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """处理特征"""
    
    # 处理用户特征
    users['yelping_days'] = (pd.to_datetime('now') - pd.to_datetime(users['yelping_since'])).dt.days
    users = users.drop('yelping_since', axis=1)
    
    # 处理商家特征
    # 将categories转换为one-hot编码
    businesses['categories'] = businesses['categories'].fillna('')
    categories = businesses['categories'].str.split(', ')
    top_categories = set()
    for cats in categories:
        if isinstance(cats, list):
            top_categories.update(cats)
    top_categories = list(top_categories)[:10]  # 只使用前10个类别
    
    for cat in top_categories:
        businesses[f'cat_{cat}'] = businesses['categories'].str.contains(cat, regex=False).astype(int)
    
    businesses = businesses.drop('categories', axis=1)
    
    return reviews, users, businesses

def split_data(
    reviews: pd.DataFrame,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    time_based: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """将数据分割为训练/验证/测试集"""
    
    if time_based:
        reviews['date'] = pd.to_datetime(reviews['date'])
        reviews = reviews.sort_values('date')
        
        n = len(reviews)
        train_idx = int(n * train_ratio)
        valid_idx = int(n * (train_ratio + valid_ratio))
        
        train = reviews.iloc[:train_idx]
        valid = reviews.iloc[train_idx:valid_idx]
        test = reviews.iloc[valid_idx:]
    else:
        train, temp = train_test_split(reviews, train_size=train_ratio, random_state=42)
        valid, test = train_test_split(temp, train_size=valid_ratio/(1-train_ratio), random_state=42)
    
    return train, valid, test

def main():
    """主预处理函数"""
    # 加载原始数据
    reviews, users, businesses = load_raw_data('data/raw')
    
    # 处理特征
    reviews, users, businesses = process_features(reviews, users, businesses)
    
    # 分割数据
    train, valid, test = split_data(reviews)
    
    # 创建输出目录
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存处理后的数据
    train.to_csv(output_dir / 'train_interactions.csv', index=False)
    valid.to_csv(output_dir / 'valid_interactions.csv', index=False)
    test.to_csv(output_dir / 'test_interactions.csv', index=False)
    users.to_csv(output_dir / 'user_features.csv', index=False)
    businesses.to_csv(output_dir / 'business_features.csv', index=False)

if __name__ == '__main__':
    main() 