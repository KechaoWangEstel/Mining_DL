# 经过测试使用model/model_best0.642_thres0.3.pth，且使用阈值为0.5时效果最好，可以达到指标如下
# Accuracy: 0.8184527868149851
# Precision: 0.8295413111645884
# Recall: 0.8304800146012046
# F1 Score: 0.83001039747542
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import ANN_relu_2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 步骤1: 读取CSV文件
df = pd.read_csv('data/normalized_mining.csv')
# 步骤2: 数据预处理
X = df.values  # 特征数据

# 步骤3: 转换数据为Tensor
X_tensor = torch.tensor(X, dtype=torch.float32)

# 步骤4: 加载模型
model = torch.load('model/model_best0.642_thres0.3.pth')  # 替换为你的模型文件路径
model = model.to(device)
model.eval()  # 设置为评估模式

# 步骤5: 创建数据加载器
test_dataset = TensorDataset(X_tensor)
# test_dataset = TensorDataset(X_tensor, y_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

threshold = 0.5            #阈值threshold
predictions = []
output_list=[]
with torch.no_grad():
    for params in test_loader:                        # 测试数据集中取数据
        params = params[0].to(device)
        outputs = model(params).reshape(-1)
        pred = (outputs > threshold).float()  # 应用阈值进行分类
        output_list.extend(outputs.view(-1).tolist())
        predictions.extend(pred.view(-1).tolist())  # 将预测结果添加到列表
df['possibilities'] = output_list
df['prediction'] = predictions
print(predictions)
df.to_csv('data/predict_result.csv', index=False, encoding='utf-8-sig')