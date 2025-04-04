import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import torch.optim as optim
from creat_data import create_data_fun
from model import ANN_relu_2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device='cpu'

# 准备数据集
test_rate = 0.2
train_data,test_data = create_data_fun('data/normalized.csv',test_rate)
# 获得数据集的长度 len(), 即length
train_data_size = len(train_data)
test_data_size = len(test_data)
# 格式化字符串, format() 中的数据会替换 {}
print("训练数据集及的长度为: {}".format(train_data_size))
print("测试数据集及的长度为: {}".format(test_data_size))

# 模型超参数
batch_size = 512
input_channels = 16  # 根据你的数据调整输入通道数
hidden_dim = 1024 #hidden layer 神经元个数
num_classes = 1    # 根据你的任务调整类别数
num_layers = 50

# 利用DataLoader 来加载数据
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = ANN_relu_2(input_channels, num_classes,hidden_dim,num_layers)
model = model.to(device)                    # 在 GPU 上进行训练
# 创建损失函数
# loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.BCELoss()
loss_fn = loss_fn.to(device)                # 在 GPU 上进行训练
# 优化器
learning_rate = 1e-4       # 1e-2 = 1 * (10)^(-2) = 1 / 100 = 0.01
# optimizer = optim.SGD(model.parameters(), lr = learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0                        # 记录训练的次数
total_test_step = 0                         # 记录测试的次数
epoch = 200                             # 训练的轮数

# 添加tensorboard
writer = SummaryWriter("logs_train")
start_time = time.time()                    # 开始训练的时间
best = 0.63
best_file = "model/model_best_thres.pth"
for i in range(epoch):
    print("------第 {} 轮训练开始------".format(i+1))

    # 训练步骤开始
    for data in train_dataloader:
        params, labels = data
        params = params.to(device)
        
        labels = labels.to(device)
        outputs = model(params).reshape(-1)              # 将训练的数据放入
        loss = loss_fn(outputs, labels)    # 得到损失值

        optimizer.zero_grad()               # 优化过程中首先要使用优化器进行梯度清零
        loss.backward()                     # 调用得到的损失，利用反向传播，得到每一个参数节点的梯度
        optimizer.step()                    # 对参数进行优化
        total_train_step += 1               # 上面就是进行了一次训练，训练次数 +1

        # 只有训练步骤是100 倍数的时候才打印数据，可以减少一些没有用的数据，方便我们找到其他数据
        if total_train_step % 100 == 0:
            end_time = time.time()          # 训练结束时间
            # print("训练时间: {}".format(end_time - start_time))
            # print("训练次数: {}, Loss: {}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)


    # 如何知道模型有没有训练好，即有咩有达到自己想要的需求
    # 我们可以在每次训练完一轮后，进行一次测试，在测试数据集上跑一遍，以测试数据集上的损失或正确率评估我们的模型有没有训练好

    # 顾名思义，下面的代码没有梯度，即我们不会利用进行调优                                   # 准确率
    threshold = 0.5             #阈值threshold
    predictions = []
    output_list=[]
    true_labels = []
    with torch.no_grad():
        for data in test_dataloader:                        # 测试数据集中取数据
            params, labels = data
            params = params.to(device)
            labels = labels.to(device)
            outputs = model(params).reshape(-1)
            loss = loss_fn(outputs, labels)                # 这里的 loss 只是一部分数据(data) 在网络模型上的损失
            pred = (outputs > threshold).float()  # 应用阈值进行分类
            predictions.extend(pred.view(-1).tolist())  # 将预测结果添加到列表
            true_labels.extend(labels.tolist())  # 将真实标签添加到列表
    # 计算性能指标
    accuracy = accuracy_score(true_labels, predictions)   # 准确率（Accuracy）
    precision = precision_score(true_labels, predictions) # 精确率（Precision）
    recall = recall_score(true_labels, predictions) #召回率（Recall）
    f1 = f1_score(true_labels, predictions)         #F1分数（F1 Score）

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    if accuracy>best:
        best = accuracy
        if os.path.exists(best_file):
            os.remove(best_file)
        torch.save(model, "model/model_best{:.3f}_thres{}.pth".format(best,threshold))
        best_file = "model/model_best{:.3f}_thres{}.pth".format(best,threshold)
    # if i %100 ==0:
    #     torch.save(model, "model/model_{}.pth".format(i))
    #     print("模型已保存")
writer.close()
