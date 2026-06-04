import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.features[idx]


class base_model(nn.Module):
    def __init__(self, input_dim=4):
        super(base_model, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim, bias=False)
        self.layer2 = nn.Linear(input_dim,input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x):
        outputs = []
        self.x1 = self.linear1(x)
        outputs.append(self.x1)
        self.x2 = torch.tanh(self.layer2(self.x1))
        outputs.append(self.x2)
        self.x3 = self.linear2(self.x2)
        outputs.append(self.x3)

        return outputs

    def regularization_loss(self, weight=1):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        lambdas = [0.1 for i in range(4)]
        cov_matrix = covariance_matrix(self.x1)
        identity_matrix = torch.eye(cov_matrix.size(0)).to(device)
        l1 = torch.norm(cov_matrix - identity_matrix, p=1)
        l2 = torch.norm(0.25 * (self.x1**4), p=1)
        W = self.linear1.weight
        WWt = W @ W.T  # 计算权重矩阵与其转置的乘积
        I = torch.eye(WWt.size(0)).to(device)
        l3 = torch.norm(WWt - I, p=1)

        cov_matrix2 = covariance_matrix(self.linear2.weight)
        mask_matrix2 = 1 - torch.eye(cov_matrix2.size(0), device=device)
        l4 = torch.norm(cov_matrix2 * mask_matrix2, p=1)
        reg_loss = (
            l1 * lambdas[0] + l2 * lambdas[1] + l3 * lambdas[2] + l4 * lambdas[3] * 2
        )
        return weight * reg_loss


def covariance_matrix(x):
    n = x.size(0)
    x = x - x.mean(dim=0, keepdim=True)  # 去中心化
    cov = (1 / (n - 1)) * x.T.matmul(x)
    return cov
