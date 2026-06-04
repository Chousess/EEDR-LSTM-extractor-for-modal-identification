import torch
import torch.nn as nn


class PerFeatureStandardize(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(PerFeatureStandardize, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # x的维度为[batch_size, seq_length, num_features]
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)

        # 标准化
        x_standardized = (x - mean) / (std + self.epsilon)
        return x_standardized
