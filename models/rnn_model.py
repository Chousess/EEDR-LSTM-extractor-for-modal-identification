import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F

def covariance_matrix(x):
    n = x.size(0)
    x = x - x.mean(dim=0, keepdim=True)  # 去中心化
    cov = (1 / (n - 1)) * x.T.matmul(x)
    return cov


# 定义模型
class CustomRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=4):
        super(CustomRNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=False)
        self.rnn = nn.RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
        )
        self.decoder=nn.Linear(hidden_size,input_size)
        # self.linearC = nn.Linear(hidden_size, output_size, bias=False)
        self.linear2 = nn.Linear(input_size, input_size, bias=False)

        # 自定义正则化项可以在这里定义
        self.custom_regularization_weight = 1

    def forward(self, x):
        outputs = []
        self.x1 = self.linear1(x)
        outputs.append(self.x1)
        self.x2, _ = self.rnn(self.x1)
        # self.x2 = self.linearC(self.xx)
        self.x2=self.decoder(self.x2)
        outputs.append(self.x2)
        self.x3 = self.linear2(self.x2)
        outputs.append(self.x3)
        return outputs

    def regularization_loss(self):
        lambdas = [0.15 for i in range(4)]
        cov_matrix = covariance_matrix(self.x1)
        identity_matrix = torch.eye(cov_matrix.size(0))
        l1 = torch.norm(cov_matrix - identity_matrix, p=1)
        l2 = torch.norm(0.25 * (self.x1**4), p=1)
        W = self.linear1.weight
        WWt = W @ W.T  # 计算权重矩阵与其转置的乘积
        I = torch.eye(WWt.size(0))
        l3 = torch.norm(WWt - I, p=1)

        cov_matrix2 = covariance_matrix(self.x2)
        identity_matrix2 = torch.eye(cov_matrix2.size(0))
        l4 = torch.norm(cov_matrix2 - identity_matrix2, p=1)
        reg_loss = l1 * lambdas[0] + l2 * lambdas[1] + l3 * lambdas[2] + l4 * lambdas[3]
        return self.custom_regularization_weight * reg_loss
