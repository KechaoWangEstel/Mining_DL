# # 1D-CNN
import torch
from torch import nn
import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self, input_channels, num_classes, hidden_dim, num_layers=9):
        super(ANN, self).__init__()
        self.num_layers = num_layers
        self.fc_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()

        # 创建重复的层
        for i in range(num_layers):
            in_features = input_channels if i == 0 else hidden_dim
            self.fc_list.append(nn.Linear(in_features, hidden_dim))
            self.bn_list.append(nn.BatchNorm1d(hidden_dim))

        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = x
        for i in range(self.num_layers-1):
            fc = self.fc_list[i]
            bn = self.bn_list[i]
            out = F.leaky_relu(bn(fc(out)))

        # 最后一层不使用激活函数
        out = self.fc_out(out)
        return out

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

class ANN_relu_2(nn.Module):
    def __init__(self, input_channels, num_classes, hidden_dim, num_layers):
        super(ANN_relu_2, self).__init__()
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
        out = self.fc_out(out)
        # 应用Sigmoid函数将输出压缩到0-1之间
        out = torch.sigmoid(out)
        return out
