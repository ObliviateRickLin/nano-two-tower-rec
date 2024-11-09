import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_raw_data(data_dir: str, sample_size: int = 100000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载Yelp原始数据集的子集
    
    Args:
        data_dir: 数据目录路径
        sample_size: 要采样的评论数量
    """
    data_dir = Path(data_dir)
    
    # 分块读取评论数据并随机采样
    chunks = pd.read_json(data_dir / 'yelp_academic_dataset_review.json', lines=True, chunksize=10000)
    sampled_reviews = []
    total_rows = 0
    
    for chunk in chunks:
        if total_rows >= sample_size:
            break
        sample_size_chunk = min(len(chunk), sample_size - total_rows)
        sampled_chunk = chunk.sample(n=sample_size_chunk)
        sampled_reviews.append(sampled_chunk)
        total_rows += sample_size_chunk
    
    reviews = pd.concat(sampled_reviews)[['user_id', 'business_id', 'stars', 'date']]
    
    # 获取采样数据中的唯一用户和商家ID
    unique_users = reviews['user_id'].unique()
    unique_businesses = reviews['business_id'].unique()
    
    # 只加载相关的用户和商家数据
    users = []
    for chunk in pd.read_json(data_dir / 'yelp_academic_dataset_user.json', lines=True, chunksize=10000):
        relevant_users = chunk[chunk['user_id'].isin(unique_users)]
        users.append(relevant_users)
    users = pd.concat(users)[['user_id', 'review_count', 'yelping_since', 'average_stars']]
    
    businesses = []
    for chunk in pd.read_json(data_dir / 'yelp_academic_dataset_business.json', lines=True, chunksize=10000):
        relevant_businesses = chunk[chunk['business_id'].isin(unique_businesses)]
        businesses.append(relevant_businesses)
    businesses = pd.concat(businesses)[['business_id', 'stars', 'review_count', 'categories']]
    
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
    # 设置较小的样本量以适应Kaggle环境
    sample_size = 100000  # 可以根据Kaggle内存调整这个数值
    
    # 设置数据路径
    data_dir = Path("../input/yelp-dataset")  # Kaggle数据集的标准路径
    output_dir = Path("../working/data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载采样后的数据
    reviews, users, businesses = load_raw_data(data_dir, sample_size=sample_size)
    print(f"加载的评论数量: {len(reviews)}")
    print(f"相关用户数量: {len(users)}")
    print(f"相关商家数量: {len(businesses)}")
    
    # 处理特征
    reviews, users, businesses = process_features(reviews, users, businesses)
    
    # 分割数据
    train, valid, test = split_data(reviews)
    
    # 保存处理后的数据
    train.to_csv(output_dir / 'train_interactions.csv', index=False)
    valid.to_csv(output_dir / 'valid_interactions.csv', index=False)
    test.to_csv(output_dir / 'test_interactions.csv', index=False)
    users.to_csv(output_dir / 'user_features.csv', index=False)
    businesses.to_csv(output_dir / 'business_features.csv', index=False)
    
    print("数据预处理完成！")
    print(f"训练集大小: {len(train)}")
    print(f"验证集大小: {len(valid)}")
    print(f"测试集大小: {len(test)}")

if __name__ == '__main__':
    main() 