import torch
import torch.nn as nn
import pandas as pd
from models.base_model1 import base_model
from rnn_model import CustomRNN
from cnn_model import CNNSequenceModel
from lstm_model import lstmModel
from GRU_model import GRuModel
from transformer_model import TransformerModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def forQMatrix(modelStr: str, num: int):
    outputs_pth = (
        "E:\ScientificResearch\SHM\code\outputs\\"
        + modelStr
        + "_output"
        + str(num)
        + ".csv"
    )
    if modelStr == "transformer":
        model = TransformerModel()
    elif modelStr == "base":
        model = base_model()
    elif modelStr == "rnn":
        model = CustomRNN()
    elif modelStr == "cnn":
        model = CNNSequenceModel()
    elif modelStr == "lstm":
        model = lstmModel()
    elif modelStr == "gru":
        model = GRuModel()
    else:
        raise ValueError("No model.")
    model.load_state_dict(
        torch.load(
            "E:\ScientificResearch\SHM\code\outputs\\"
            + modelStr
            + "_best_model"
            + str(num)
            + ".pth"
        )
    )
    model.eval()  # 将模型设置为评估模式
    # 读取CSV文件
    df = pd.read_csv("E:\ScientificResearch\SHM\code\\data_noise.csv")
    X = torch.tensor(df.values, dtype=torch.float)
    with torch.no_grad():  # 确保在这个过程中不计算梯度
        outputs = model(X)  # 获取第二层的输出
        Q = outputs[1]

        # 步骤2: 读取CSV文件
    df = pd.read_csv("../data_noise.csv")  # 替换 'input.csv' 为你的文件路径
    # 将DataFrame转换为numpy数组进行矩阵乘法
    input_matrix = df.values
    matrix_4x4 = np.matrix(model.linear2.weight.data.numpy())
    # 步骤3: 矩阵相乘
    result_matrix = np.dot(input_matrix, matrix_4x4.I)

    # 步骤4: 将结果写入新的CSV文件
    result_df = pd.DataFrame(result_matrix)
    result_df.to_csv("baseline.csv", index=False)  # 'output.csv' 是输出文件的名称

    data = result_df
    time = np.linspace(0, 10, len(data))
    # 为每个自由度计算加速度
    accelerations = []
    for i in range(4):  # 遍历所有列
        print(i)
        aa = data.iloc[:, i]
        accelerations.append(aa)
        # 绘制每个自由度的加速度图像
        plt.figure(figsize=(10, 6))  # 设置图像大小
        plt.plot(time, aa, label=f"Acceleration for Freedom {i+1}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Acceleration")
        plt.title(f"Acceleration vs. Time for Freedom {i+1}")
        plt.legend()
        plt.grid(True)

    plt.show()


if __name__ == "__main__":
    num = 9
    modelStr = "lstm"
    forQMatrix(modelStr, num)
