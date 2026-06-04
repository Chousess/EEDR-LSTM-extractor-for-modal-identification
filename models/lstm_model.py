import torch
import torch.nn as nn
import torch.nn.functional as F
from .standard import PerFeatureStandardize
import numpy as np

def covariance_matrix(x):
    n = x.size(0)
    x = x - x.mean(dim=0, keepdim=True)  # 去中心化
    cov = (1 / (n - 1)) * x.T.matmul(x)
    return cov

def cal_correlation_matrix(x):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 将数据转换为CPU上的NumPy数组
    x_cpu = x.detach().cpu().numpy()
    # 计算相关性系数矩阵
    correlation_matrix = np.corrcoef(x_cpu, rowvar=False)
    # 将相关性系数矩阵重新转换回GPU上的张量
    correlation_matrix_gpu = torch.tensor(correlation_matrix, device=device)
    return correlation_matrix_gpu

class lstmModel(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=128, lstm_dim=64, output_dim=4):
        super(lstmModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm_dim = lstm_dim
        self.output_dim = output_dim

        self.linear1 = nn.Linear(input_dim, input_dim, bias=False)
        self.encoder_linear = nn.Linear(input_dim, hidden_dim)
        #self.standardize = PerFeatureStandardize()
        self.lstm = nn.LSTM(hidden_dim, lstm_dim, batch_first=True)
        self.decoder_linear = nn.Linear(lstm_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)

    def forward(self, x):
        # x: [batch_size, sequence_length, input_dim]
        # Linear layer
        outputs = []
        #self.x1 = self.linear1(x)
        outputs.append(x)
        #x = F.tanh(self.encoder_linear(x))
        x = self.encoder_linear(x)
        #x = self.standardize(x)
        # LSTM layer
        lstm_out, (hn, cn) = self.lstm(x)
        # Linear layer to decode
        self.x2 = self.decoder_linear(lstm_out).squeeze(0)
        # self.x2 = (lstm_out).squeeze(0)
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

        #cov_matrix2 = covariance_matrix(self.x2)
        #mask_matrix2 = 1 - torch.eye(cov_matrix2.size(0), device=device)
        #t= cov_matrix2 * mask_matrix2
        cov_matrix2 = cal_correlation_matrix(self.x2)
        #cov_matrix2 = covariance_matrix(self.x2)
        identity_matrix2 = torch.eye(cov_matrix2.size(0)).to(device)
        l4 = torch.norm(cov_matrix2 - identity_matrix2, p=1)
        print("l1: "+str(l1))
        print("l2: "+str(l2))
        print("l3: "+str(l3))
        print("l4: "+str(l4))
        reg_loss = (
            l1 * lambdas[0] + 0.1*l2 * lambdas[1] + l3 * lambdas[2] + l4 * lambdas[3]
        )
        return weight * reg_loss
