import torch
import numpy as np
from scipy.signal import welch, find_peaks

def compute_covariance_matrix(x):
    n = x.size(0)
    x = x - x.mean(dim=0, keepdim=True)  # 去中心化
    cov = (1 / (n - 1)) * x.T.matmul(x)
    return cov


def covariance_penalty(output):
   # print(output.shape)
    covariance_matrix = compute_covariance_matrix(output)
    # 计算非对角元素的平方和
    # 使用mask来选取非对角元素
    identity = torch.eye(covariance_matrix.size(-1), device=output.device)
    non_diag_mask = 1 - identity
    non_diag_elements = covariance_matrix * non_diag_mask
    penalty = torch.sum(non_diag_elements**2)
    return penalty


def cal_correlation_matrix(x):
    device = x.device
    # 将数据转换为CPU上的NumPy数组
    x_cpu = x.detach().cpu().numpy()
    # 计算相关性系数矩阵
    correlation_matrix = np.corrcoef(x_cpu, rowvar=False)
    # 将相关性系数矩阵重新转换回GPU上的张量
    correlation_matrix_gpu = torch.tensor(correlation_matrix, device=device)
    return correlation_matrix_gpu


def correlation_penalty(output):
    correlation_matrix = cal_correlation_matrix(output)
    identity = torch.eye(correlation_matrix.size(-1), device=correlation_matrix.device)
    non_diag_mask = 1 - identity
    non_diag_elements = correlation_matrix * non_diag_mask
    #penalty = torch.sum(non_diag_elements**2)
    penalty = torch.sum(torch.abs(non_diag_elements))
    return penalty

def non_gaussian_penalty(outputs):
    # 峰度计算，outputs 的形状应为 [sequence_length, feature_dim]
    kurtosis = 0
    for i in range(outputs.shape[1]):  # 遍历每个特征维度
        mean_i = torch.mean(outputs[:, i])
        std_i = torch.std(outputs[:, i])
        kurtosis += torch.mean((outputs[:, i] - mean_i) ** 4) / (std_i**4)
    kurtosis /= outputs.shape[1]  # 计算平均峰度
    return kurtosis


def negentropy_loss(self, Y):
        # 数据预处理：去均值和标准化
        Y_centered = Y - Y.mean(dim=0)
        Y_normalized = Y_centered / (Y.std(dim=0) + 1e-9)  # 加1e-9以避免除以零
        
        # 使用对比函数 G(y) = 1/4 * y^4 来计算负熵的近似值
        G = (1/4) * torch.pow(Y_normalized, 4)
        gaussian_entropy = 0.5 * (1 + torch.log(2 * torch.pi * torch.var(Y_normalized, dim=0) + 1e-9))
        actual_entropy = torch.mean(G, dim=0)
        negentropy = gaussian_entropy - actual_entropy
        negentropy_loss = torch.sum(negentropy)
        return negentropy_loss

def covariance_loss(Y):
    # 计算协方差矩阵
    Y = Y - Y.mean(dim=0)
    covariance_matrix = torch.mm(Y.T, Y) / (Y.size(0) - 1)
    identity_matrix = torch.eye(covariance_matrix.size(0)).to(covariance_matrix.device)
    covariance_loss = torch.norm(covariance_matrix - identity_matrix, p='fro')
    return covariance_loss



