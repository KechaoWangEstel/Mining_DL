import torch
from torch.utils.data import Dataset, DataLoader,Subset,random_split
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class PointsDataset(Dataset):
    def __init__(self, df, features, label):
        """
        参数:
        - df: DataFrame，包含数据集的所有数据。
        - features: 特征列的名称列表。
        - label: 标签列的名称。
        """
        self.features = features
        self.label = label
        self.data = torch.tensor(df[features].values, dtype=torch.float32)
        self.labels = torch.tensor(df[label].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_data_fun(file_name,test_rate):
    '''
    filename:数据文件名
    确保数据文件中，除了label列，其他列都是训练参数
    test_rate:测试集的比例
    '''
    # 读取CSV文件到DataFrame
    df = pd.read_csv(file_name)
    cols_to_scale = df.columns.tolist()
    cols_to_scale.remove('label')
    feature_columns = cols_to_scale
    label_column = 'label'
    sampled_df = df.sample(frac=1)  #采样100%

    # 拆分数据集为训练集和测试集
    train_df, test_df = train_test_split(sampled_df, test_size=test_rate, random_state=42)
    
    # 创建训练集和测试集的PyTorch数据集
    train_dataset = PointsDataset(train_df, feature_columns, label_column)
    test_dataset = PointsDataset(test_df, feature_columns, label_column)
    return train_dataset,test_dataset



if __name__ =='__main__':
    train_dataset,test_dataset = create_data_fun('data/normalized.csv',0.2)
    # 创建DataLoader来批量加载数据
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # 现在你可以使用DataLoader来迭代你的数据集
    for batch_idx, (data, labels) in enumerate(train_loader):
        # 在这里使用你的模型和数据
        pass
