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
from models.base_model1 import CustomDataset, base_model
from models.rnn_model import CustomRNN
from models.cnn_model import CNNSequenceModel
from models.lstm_model import lstmModel
from models.GRU_model import GRuModel
from models.transformer_model import TransformerModel
from models.lstm_tsfm_model import TransformerAutoencoder
from models.lstm_RsNet import BlindSourceSeparator
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
import sys
from reg import *

fileStr = r"D:\\user\\xxh\\研究or项目\\LSTM\\LSTM汇总\\代码\\实验matlab\\experiment\\experiment_5.csv"  # 换训练数据时注意改文件名
modelStr = "transformer"
num = 4
logStr = "experiment_logs\\" + modelStr + "_" + str(num) + "_log.txt"
myEpoch = 1000
input_dim = 4
batch_size = 128
selected=np.array([1,2,3,5,6])-1

#设置随机数种子
def set_seed(seed_value=0):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module. #初始化随机数，每次运行生成的随机数相同
    np.random.seed(seed_value)  # Numpy module.
    torch.manual_seed(seed_value)  # PyTorch random number generator.
    # if you are using multi-GPU, set the seed for all GPUs (it's better to avoid using multi-GPU with random processes because it's hard to reproduce)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)

#写入文件
def log(message, log_file=logStr):
    print(message)  # 控制台打印信息
    with open(log_file, "a") as file:  # 打开文件追加模式
        file.write(message + "\n")  # 写入信息到文件并添加换行


def main(modelStr, num):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #设置使用的设备
    set_seed(0)
    print(f"Using device: {device}")
    df = pd.read_csv(fileStr)       #读取训练用的数据集
    #data = df.iloc[:,selected]
    data=df.values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    #data_scaled=np.hstack([data_scaled]*2)
    X = torch.tensor(data_scaled, dtype=torch.float).to(device)    # tensor用于数据的存储和变换，类似
    dataset = TensorDataset(X, X)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if modelStr == "base":
        model = base_model(input_dim=input_dim)
        model_eval=base_model(input_dim=input_dim)
    elif modelStr == "rnn":
        model = CustomRNN(input_size=input_dim,hidden_size=64)
        model_eval=CustomRNN(input_size=input_dim,hidden_size=64)
    elif modelStr == "cnn":
        model = CNNSequenceModel()
    elif modelStr == "lstm":
        model = lstmModel(
            input_dim=input_dim, hidden_dim=32, lstm_dim=32, output_dim=input_dim
        )
        model_eval=lstmModel(
            input_dim=input_dim, hidden_dim=32, lstm_dim=32, output_dim=input_dim
        )
        # model = lstmModel()
    elif modelStr=="lstm_rsnet":
        model=BlindSourceSeparator(input_dim=input_dim,hidden_dim=64)
        model_eval=BlindSourceSeparator(input_dim=input_dim,hidden_dim=64)
    elif modelStr == "gru":
        model = GRuModel(input_dim=input_dim,hidden_dim=64,gru_dim=64,output_dim=input_dim)
        model_eval = GRuModel(input_dim=input_dim,hidden_dim=64,gru_dim=64,output_dim=input_dim)
    elif modelStr == "transformer":
        model = TransformerModel(input_dim=input_dim)
        model_eval = TransformerModel(input_dim=input_dim)
    elif modelStr == "lstm_tsfm":
        model = TransformerAutoencoder(
            input_dim=17,
            d_model=128,
            nhead=8,
            num_encoder_layers=6,
            lstm_hidden_dim=64,
            num_lstm_layers=1,
            output_dim=17,
        )
    else:
        raise ValueError("No model.")
    model = model.to(device)
    log(fileStr)
    log(str(model))
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    criterion = nn.MSELoss().to(device)
    losses = []
    min_loss = float("inf")
    ResultString = ""
    num_epochs = myEpoch
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # print(inputs-targets)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss1 = criterion(outputs[2], targets)
            loss2 = (
                non_gaussian_penalty(outputs[1]) * 0.1
                + covariance_penalty(outputs[1]) * 0.1
                #+kurtosis_loss(outputs[1])*0.1
                #+negentropy_loss(outputs[1])*0.1
                #+covariance_loss(outputs[1])*0.1
            )
            #log("loss:" + str(loss1))
            #log("reg:" + str(loss2))
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        log(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")
        if avg_loss < min_loss:
            model = model.to("cpu")
            min_loss = avg_loss
            torch.save(
                model.state_dict(),
                "experiment_outputs/" + modelStr + str(num) + ".pth",
            )
            model = model.to(device)
            ResultString = "epoch:" + str(epoch + 1) + ", minimum loss:" + str(min_loss)

    log(ResultString)
    model.to("cpu")
    log(str(model.linear2.weight.data))
    outputs_pth = (
        "experiment_outputs\\"
        + modelStr
        + "_output"
        + str(num)
        + ".csv"
    )
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show(block=False)

    model_eval.load_state_dict(
        torch.load(
            "experiment_outputs\\"
            + modelStr
            + str(num)
            + ".pth"
        )
    )
    model_eval.eval()  # 将模型设置为评估模式

    X2 = torch.tensor(data_scaled, dtype=torch.float)
    with torch.no_grad():  # 确保在这个过程中不计算梯度
        outputs = model_eval(X2)  # 获取第二层的输出
        Q = outputs[1]
    Q_output = pd.DataFrame(Q.numpy())
    Q_output.to_csv(outputs_pth, index=False,header=None)

    pre=0
    data = Q_output
    time = np.linspace(0, 1, len(data)-pre)
    # 为每个自由度计算响应
    response = []
    for i in range(Q_output.shape[1]):  # 遍历所有列
        print(i)
        aa = data.iloc[pre:, i]
        #aa = data.iloc[5000:6000, i]
        response.append(aa)
        plt.figure(figsize=(10, 6))  # 设置图像大小
        plt.plot(time, aa, label=f"Freedom {i+1}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("")
        plt.title(f"Response vs. Time for Freedom {i+1}")
        plt.legend()
        plt.grid(True)

    plt.show()

if __name__ == "__main__":

    main(modelStr, num)
