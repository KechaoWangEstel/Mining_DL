import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
from model import ANN_relu_2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 步骤1: 读取CSV文件
df = pd.read_csv('data/normalized.csv')

# 步骤2: 数据预处理
# 假设特征列不需要预处理，或者已经预处理完毕
X = df.iloc[:, :-1].values  # 特征数据
y = df['label'].values  # 标签数据

# 步骤3: 转换数据为Tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)  # 标签数据类型转换

# 步骤4: 加载模型
model = torch.load('model/model_best_0.672_50_1024.pth')  # 替换为你的模型文件路径
model = model.to(device)
model.eval()  # 设置为评估模式

# 步骤5: 创建数据加载器
test_dataset = TensorDataset(X_tensor, y_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 步骤6: 模型评估并计算准确率
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device),labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test data: {accuracy:.2f}%')

# 注意：这里假设模型的输出和标签数据都是整数类型，如果是多分类问题，可能需要使用softmax函数处理模型输出。