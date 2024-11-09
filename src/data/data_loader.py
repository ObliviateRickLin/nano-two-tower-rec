from typing import Dict
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class YelpDataset(Dataset):
    """Yelp推荐数据集类"""
    
    def __init__(self, data_path: str, mode: str = 'train'):
        """
        参数:
            data_path: 处理后数据目录的路径
            mode: 'train', 'valid', 或 'test' 之一
        """
        self.mode = mode
        self.data = self._load_data(data_path)
        self._process_data()
        
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """加载并预处理数据"""
        # 加载交互数据
        df = pd.read_csv(f'{data_path}/{self.mode}_interactions.csv')
        
        # 加载用户和商家特征
        user_features = pd.read_csv(f'{data_path}/user_features.csv')
        business_features = pd.read_csv(f'{data_path}/business_features.csv')
        
        # 合并特征
        df = df.merge(user_features, on='user_id', how='left')
        df = df.merge(business_features, on='business_id', how='left')
        
        return df
        
    def _process_data(self):
        """处理模型输入数据"""
        # 编码用户和商家ID
        if 'user_idx' not in self.data.columns:
            user_encoder = LabelEncoder()
            self.data['user_idx'] = user_encoder.fit_transform(self.data['user_id'])
        
        if 'business_idx' not in self.data.columns:
            business_encoder = LabelEncoder()
            self.data['business_idx'] = business_encoder.fit_transform(self.data['business_id'])
            
        # 创建交互矩阵用于负采样
        if self.mode == 'train':
            self.interaction_matrix = self._create_interaction_matrix()
            
    def _create_interaction_matrix(self) -> torch.Tensor:
        """创建用于负采样的稀疏交互矩阵"""
        num_users = self.data['user_idx'].nunique()
        num_businesses = self.data['business_idx'].nunique()
        
        interactions = torch.zeros((num_users, num_businesses), dtype=torch.bool)
        for _, row in self.data.iterrows():
            interactions[row['user_idx'], row['business_idx']] = 1
            
        return interactions

    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        
        # Features
        features = {
            # 用户特征
            'user_features': torch.tensor([
                row['user_idx'],  # 用户ID编码
                row['review_count_x'],  # 用户评论数
                row['average_stars'],  # 用户平均评分
                row['yelping_days']  # 用户注册天数
            ], dtype=torch.float),
            
            # 商家特征
            'business_features': torch.tensor([
                row['business_idx'],  # 商家ID编码
                row['review_count_y'],  # 商家评论数
                row['stars_y']  # 商家平均评分
            ], dtype=torch.float),
            
            # 商家类别特征 (one-hot编码)
            'category_features': torch.tensor([
                row[col] for col in row.index if col.startswith('cat_')
            ], dtype=torch.float)
        }
        
        # Label
        label = torch.tensor(row['stars_x'], dtype=torch.float)
        
        return features, label

def get_dataloader(
    data_path: str,
    batch_size: int,
    mode: str = 'train',
    num_workers: int = 4
) -> DataLoader:
    """创建数据加载器"""
    dataset = YelpDataset(data_path, mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True
    ) 