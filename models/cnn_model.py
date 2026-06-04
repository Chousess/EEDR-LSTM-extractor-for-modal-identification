import torch
import torch.nn as nn
import torch.nn.functional as F


def covariance_matrix(x):
    n = x.size(0)
    x = x - x.mean(dim=0, keepdim=True)  # 去中心化
    cov = (1 / (n - 1)) * x.T.matmul(x)
    return cov


class CNNSequenceModel(nn.Module):
    def __init__(self, input_size=4, sequence_length=1000, output_size=4):
        super(CNNSequenceModel, self).__init__()
        # 初始线性层
        self.linear1 = nn.Linear(
            input_size, input_size, bias=False
        )  # 假设增加维度以适配卷积
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=4, out_channels=16, kernel_size=5, stride=1, padding=2
            ),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            # nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                in_channels=16, out_channels=4, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
        )
        self.linear2 = nn.Linear(output_size, output_size, bias=False)

    def forward(self, x):
        outputs = []
        self.x1 = self.linear1(x)
        outputs.append(self.x1)
        x = self.x1.transpose(0, 1).unsqueeze(0)  # Reshape for Conv1d
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        self.x2 = decoded.squeeze(0).transpose(0, 1)
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
