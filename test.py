import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ANN_relu(nn.Module):
    def __init__(self, input_channels, num_classes, hidden_dim, num_layers):
        super(ANN_relu, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_channels = input_channels
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # 创建重复的层
        for i in range(num_layers):
            in_features = input_channels if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_features, hidden_dim))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

        # 最后一层的输出维度与类别数相同
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = x
        fc = self.layers[0]
        bn = self.bn_layers[0]
        out = F.leaky_relu(bn(fc(out)))

        for i in range(1,self.num_layers):
            fc = self.layers[i]
            bn = self.bn_layers[i]
            out = F.leaky_relu(bn(fc(out) + out))  # 残差连接
            out = torch.relu(out)  # 这里使用ReLU，你可以根据需要使用其他激活函数

        # 最后一层不使用激活函数
        out = self.fc_out(bn(out))
        return out

# 步骤1: 读取CSV文件
df = pd.read_csv('normalized_mining.csv')

# 步骤2: 数据预处理
X = df.values  # 特征数据

# 步骤3: 转换数据为Tensor
X_tensor = torch.tensor(X, dtype=torch.float32)

# 步骤4: 加载模型
model = torch.load('model/model_best_0.662_50_1024.pth')  # 替换为你的模型文件路径
model = model.to(device)
model.eval()  # 设置为评估模式

# 步骤5: 创建数据加载器
test_dataset = TensorDataset(X_tensor)
# test_dataset = TensorDataset(X_tensor, y_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 步骤6: 模型评估并计算准确率
prediction_list = []
result_list = []
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs[0].to(device)
        outputs = model(inputs)
        prediction_list+=outputs.tolist()
        # _, predicted = torch.max(outputs.data, 1)
        # result_list.append(predicted)
df['prediction'] = prediction_list
df.to_csv('predict_result.csv', index=False, encoding='utf-8-sig')
# accuracy = 100 * correct / total
# print(f'Accuracy of the model on the test data: {accuracy:.2f}%')

# 注意：这里假设模型的输出和标签数据都是整数类型，如果是多分类问题，可能需要使用softmax函数处理模型输出。