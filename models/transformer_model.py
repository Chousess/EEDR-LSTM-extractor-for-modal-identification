import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def covariance_matrix(x):
    n = x.size(0)
    x = x - x.mean(dim=0, keepdim=True)  # 去中心化
    cov = (1 / (n - 1)) * x.T.matmul(x)
    return cov


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim_feedforward=2048,
        max_seq_length=100000,
        d_model=32,
        input_dim=4,
        nhead=4,
        num_encoder_layers=3,
    ):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(input_dim, input_dim, bias=False)
        self.embed = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_encoder_layers
        )
        self.decoder = nn.Linear(d_model,12)
        self.linear2 = nn.Linear(12,input_dim, bias=False)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        outputs = []
        self.x1 = self.linear1(x)
        outputs.append(self.x1)
        # 添加序列维度 (batch, features) -> (1, batch, features)
        x = self.x1.unsqueeze(0)
        # 嵌入并应用位置编码
        x = self.embed(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        # Transformer编码
        x = self.transformer_encoder(x)
        # 解码并移除序列维度 (1, batch, features) -> (batch, features)
        self.x2 = self.decoder(x).squeeze(0)
        outputs.append(self.x2)
        self.x3 = self.linear2(self.x2)
        outputs.append(self.x3)
        return outputs

    def regularization_loss(self, weight=1):
        device = self.x1.device
        lambdas = [0.2 for i in range(4)]
        cov_matrix = covariance_matrix(self.x1)
        identity_matrix = torch.eye(cov_matrix.size(0), device=device)
        l1 = torch.norm(cov_matrix - identity_matrix, p=1)
        l2 = torch.norm(0.25 * (self.x1**4), p=1)
        W = self.linear1.weight
        WWt = W @ W.T  # 计算权重矩阵与其转置的乘积
        I = torch.eye(WWt.size(0), device=device)
        l3 = torch.norm(WWt - I, p=1)

        cov_matrix2 = covariance_matrix(self.x2)
        identity_matrix2 = torch.eye(cov_matrix2.size(0), device=device)
        l4 = torch.norm(cov_matrix2 - identity_matrix2, p=1)
        reg_loss = l1 * lambdas[0] + l2 * lambdas[1] + l3 * lambdas[2] + l4 * lambdas[3]
        return weight * reg_loss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x
