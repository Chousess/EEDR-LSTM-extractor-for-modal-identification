import torch
import torch.nn as nn
import torch.optim as optim


def covariance_matrix(x):
    n = x.size(0)
    x = x - x.mean(dim=0, keepdim=True)  # 去中心化
    cov = (1 / (n - 1)) * x.T.matmul(x)
    return cov


import torch
import torch.nn as nn
import torch.nn.functional as F


class GRuModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, gru_dim=16, output_dim=4):
        super(GRuModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru_dim = gru_dim
        self.output_dim = output_dim

        self.linear1 = nn.Linear(input_dim,hidden_dim)
        self.encoder_linear = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder_linear = nn.Linear(gru_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: [batch_size, sequence_length, input_dim]
        # Linear layer
        outputs = []
        self.x1 = self.linear1(x)
        outputs.append(self.x1)
        x = self.x1.unsqueeze(0)
        #x = F.tanh(self.encoder_linear(x))
        # GRU layer
        gru_out, hn = self.gru(x)
        # Linear layer to decode
        self.x2 = self.decoder_linear(gru_out).squeeze(0)
        # self.x2 = (gru_out).squeeze(0)
        outputs.append(self.x2)
        self.x3 = self.linear2(self.x2)
        outputs.append(self.x3)
        return outputs

    def regularization_loss(self, weight=1):
        lambdas = [0.2 for i in range(4)]
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
        return weight * reg_loss
