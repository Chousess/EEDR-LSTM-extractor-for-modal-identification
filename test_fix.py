#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from models.transformer_model import TransformerModel

print('Testing train_flat.py data loading and model initialization...')

# 模拟train_flat.py的数据加载过程
fileStr = r'D:\\user\\xxh\\研究or项目\\LSTM\\LSTM汇总\\代码\\实验matlab\\flat\\flat_with_sequence\\data\\flat_selected.csv'
input_dim = 17+2

try:
    # 加载数据
    df = pd.read_csv(fileStr)
    selected_data = df.values
    print(f'Data loaded: {selected_data.shape}')

    # 数据预处理
    first_column = selected_data[:, 0:1]
    last_column = selected_data[:, -1:]
    data_extended = np.hstack([first_column, selected_data, last_column])
    print(f'Data extended: {data_extended.shape}')

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_extended)
    print(f'Data scaled: {data_scaled.shape}')

    # 创建模型
    model = TransformerModel(input_dim=input_dim)
    print(f'Model created with input_dim={input_dim}')

    # 测试前向传播
    X = torch.tensor(data_scaled[:128], dtype=torch.float)  # 取前128个样本
    outputs = model(X)
    print(f'Forward pass successful')
    print(f'Input shape: {X.shape}')
    print(f'Output shapes: {[o.shape for o in outputs]}')

    print('All tests passed! TransformerModel should work in train_flat.py')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()