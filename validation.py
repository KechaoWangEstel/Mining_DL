import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import ANN_relu_2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def get_validation(threshold,model,device):
    predictions = []
    true_labels = []
    output_list=[]
    with torch.no_grad():
        for data in test_loader:                        # 测试数据集中取数据
            params, labels = data
            params = params.to(device)
            labels = labels.to(device)
            outputs = model(params).reshape(-1)
            pred = (outputs > threshold).float()  # 应用阈值进行分类
            predictions.extend(pred.view(-1).tolist())  # 将预测结果添加到列表
            true_labels.extend(labels.tolist())  # 将真实标签添加到列表
            output_list.extend(outputs.view(-1).tolist())
    return true_labels,predictions,output_list


# 步骤1: 读取CSV文件
df = pd.read_csv('data/normalized.csv')

# 步骤2: 数据预处理
# 假设特征列不需要预处理，或者已经预处理完毕
X = df.iloc[:, :-1].values  # 特征数据
y = df['label'].values  # 标签数据

# 步骤3: 转换数据为Tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)  # 标签数据类型转换

# 步骤4: 加载模型
model = torch.load('model\model_best0.642_thres0.3.pth')  # 替换为你的模型文件路径
model = model.to(device)
model.eval()  # 设置为评估模式

# 步骤5: 创建数据加载器
test_dataset = TensorDataset(X_tensor, y_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# for i in range(3,9):
#     true_labels,predictions = get_validation(i/10,model,device)
#     # 计算性能指标
#     accuracy = accuracy_score(true_labels, predictions)   # 准确率（Accuracy）
#     precision = precision_score(true_labels, predictions) # 精确率（Precision）
#     recall = recall_score(true_labels, predictions) #召回率（Recall）
#     f1 = f1_score(true_labels, predictions)         #F1分数（F1 Score）
#     print(f"------threshold:{i/10}------")
#     print(f'Accuracy: {accuracy}')
#     print(f'Precision: {precision}')
#     print(f'Recall: {recall}')
#     print(f'F1 Score: {f1}')


threshold = 0.5            #阈值threshold
true_labels,predictions,output_list = get_validation(threshold,model,device)
# 计算ROC曲线的FPR和TPR
FPR, TPR, _ = roc_curve(true_labels, output_list)
# 计算AUC
roc_auc = auc(FPR, TPR)
# 绘制ROC曲线
plt.figure()
plt.plot(FPR, TPR, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
plt.show()

