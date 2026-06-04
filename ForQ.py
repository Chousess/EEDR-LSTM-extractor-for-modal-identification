import torch
import torch.nn as nn
import pandas as pd

# from .base_model1 import base_model
from models.rnn_model import CustomRNN
from models.bi_rnn_model import CustomBi_RNN
from models.cnn_model import CNNSequenceModel
from models.lstm_model import lstmModel
from models.GRU_model import GRuModel
from models.transformer_model import TransformerModel
from models.base_model1 import CustomDataset, base_model
from models.lstm_tsfm_model import TransformerAutoencoder
from models.lstm_new import LSTMAutoencoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
num = 2
input_dim=17
sensorNum=5
sensorIndex = np.array([17, 7, 8, 16, 5, 6, 12, 10, 3, 14, 4, 15, 9, 2, 11, 1, 13]) - 1
sensorIndex2= np.array([5,7,9,12,15]) - 1
modelStr="gru"
fileStr = "E:\GYJ\SHM\shm\\truss17_g1.csv"
outputs_pth = (
    "E:\GYJ\SHM\shm\\5outputs\\"
    + modelStr
    + "_output"
    + str(num)
    + ".csv"
)
if modelStr == "transformer":
    model = TransformerModel()
elif modelStr == "base":
    model = base_model(input_dim=sensorNum)
elif modelStr == "rnn":
    model = CustomRNN(input_size=sensorNum*3,hidden_size=128)
elif modelStr == "bi_rnn":
    #model = CustomRNN()
    model = CustomBi_RNN(
        input_size=sensorNum*3, hidden_size=128, output_size=sensorNum*3
    )
elif modelStr == "cnn":
    model = CNNSequenceModel()
elif modelStr == "lstm":
    model = lstmModel(input_dim=sensorNum*3, hidden_dim=128, lstm_dim=128, output_dim=sensorNum*3)
elif modelStr == "lstm2":
    model = LSTMAutoencoder(
        input_dim=sensorNum, hidden_dim=256
    )
elif modelStr == "gru":
    model = GRuModel(input_dim=sensorNum*3,hidden_dim=128,gru_dim=128,output_dim=sensorNum*3)
elif modelStr == "lstm_tsfm":
    model = TransformerAutoencoder(
                input_dim=sensorNum,
                hidden_dim=128,
                num_heads=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
            )
else:
    raise ValueError("No model.")
model.load_state_dict(
    torch.load(
        "E:\GYJ\SHM\shm\\5outputs\\"
        + modelStr
        + "_best_model"
        + str(num)
        + ".pth"
    )
)
model.eval()  # 将模型设置为评估模式
# 读取CSV文件
df = pd.read_csv(fileStr)
    # 在数据集中选择对应的列
selected_data = df.iloc[:, sensorIndex2]
scaler = StandardScaler()
selected_data = scaler.fit_transform(selected_data)
selected_data=np.hstack([selected_data] * 3)
X = torch.tensor(selected_data, dtype=torch.float)
with torch.no_grad():  # 确保在这个过程中不计算梯度
    outputs = model(X)  # 获取第二层的输出
    Q = outputs[1]
Q_output = pd.DataFrame(Q.numpy())
Q_output.to_csv(outputs_pth, index=False,header=None)

pre=0
data = Q_output
time = np.linspace(0, 1, len(data)-pre)
#time = np.linspace(0, 1, 1000)
print(len(time))
# 为每个自由度计算加速度
accelerations = []
for i in range(17):  # 遍历所有列
    print(i)
    aa = data.iloc[pre:, i]
    #aa = data.iloc[5000:6000, i]
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
