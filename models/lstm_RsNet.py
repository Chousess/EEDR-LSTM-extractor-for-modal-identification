import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out += residual  # 残差连接
        out = self.relu(out)
        return out


# 定义整个盲源分离模型
class BlindSourceSeparator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BlindSourceSeparator, self).__init__()

        # Encoder: 一个线性层加一个双向LSTM
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        )

        # Decoder: 两个残差块
        self.res_block1 = ResidualBlock(
            hidden_dim * 2
        )  # 因为双向LSTM，所以是 hidden_dim * 2
        self.res_block2 = ResidualBlock(hidden_dim * 2)
        self.decoderLinear = nn.Linear(hidden_dim * 2, input_dim)
        # 输出层
        self.linear2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        outputs = []
        x = self.linear1(x)
        outputs.append(x)
        x = x.unsqueeze(0)
        x, _ = self.lstm(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        self.x2 = self.decoderLinear(x).squeeze(0)
        outputs.append(self.x2)
        self.x3 = self.linear2(self.x2)
        outputs.append(self.x3)
        return outputs
