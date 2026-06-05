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
from models.bi_rnn_model import CustomBi_RNN
from models.cnn_model import CNNSequenceModel
from models.lstm_model import lstmModel
from models.GRU_model import GRuModel
from models.transformer_model import TransformerModel
from models.lstm_tsfm_model import TransformerAutoencoder
from models.lstm_new import LSTMAutoencoder
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
import sys
from reg import non_gaussian_penalty, correlation_penalty, covariance_penalty

sensorIndex = np.array([17, 7, 8, 16, 5, 6, 12, 10, 3, 14, 4, 15, 9, 2, 11, 1, 13]) - 1
sensorIndex2 = np.array([5, 7, 9, 12, 15]) - 1
sensorNum = 5


fileStr = r"D:\\user\\xxh\\研究or项目\\LSTM\\LSTM汇总\\代码\\实验matlab\\spring truss\\truss_17.csv"  # 换训练数据时注意改文件名
modelStr = "transformer"
num = 17
logStr = "truss_logs\\" + modelStr + "_" + str(num) + "_log.txt"
myEpoch = 1000
input_dim = 17
batch_size = 128


def load_weights_from_csv(file_path, layer):
    # Load weights from csv file
    weights_df = pd.read_csv(file_path, header=None)
    weights_matrix = weights_df.values
    weights_tensor = torch.tensor(weights_matrix, dtype=torch.float).transpose(0, 1)

    # Assign the weights to the layer's weight parameter and freeze the layer
    with torch.no_grad():
        layer.weight = nn.Parameter(weights_tensor)
    layer.weight.requires_grad = False


def set_seed(seed_value=0):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module.
    np.random.seed(seed_value)  # Numpy module.
    torch.manual_seed(seed_value)  # PyTorch random number generator.
    # if you are using multi-GPU, set the seed for all GPUs (it's better to avoid using multi-GPU with random processes because it's hard to reproduce)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)


def log(message, log_file=logStr):

    print(message)  # 控制台打印信息
    with open(log_file, "a") as file:  # 打开文件追加模式
        file.write(message + "\n")  # 写入信息到文件并添加换行


def main(modelStr, num):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(0)
    print(f"Using device: {device}")
    df = pd.read_csv(fileStr)
    # df = pd.read_csv("data_noise.csv")
    datasetSca = df.values
    selected_indices = sensorIndex2[:sensorNum]
    # 在数据集中选择对应的列
    selected_data = df.iloc[:, selected_indices]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(selected_data)
    data_scaled = np.hstack([data_scaled] * 3)
    X = torch.tensor(data_scaled, dtype=torch.float).to(device)
    # print(X)
    # dataset = CustomDataset(X)
    dataset = TensorDataset(X, X)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if modelStr == "transformer":
        model = TransformerModel(input_dim=sensorNum * 3)
        model_eval = TransformerModel(input_dim=sensorNum * 3)
    elif modelStr == "base":
        model = base_model(input_dim=sensorNum * 3)
    elif modelStr == "rnn":
        model = CustomRNN(input_size=sensorNum * 3, hidden_size=128)
    elif modelStr == "bi_rnn":
        # model = CustomRNN()
        model = CustomBi_RNN(
            input_size=sensorNum * 3, hidden_size=128, output_size=sensorNum * 3
        )
    elif modelStr == "cnn":
        model = CNNSequenceModel()
    elif modelStr == "lstm2":
        model = LSTMAutoencoder(input_dim=sensorNum, hidden_dim=256)
    elif modelStr == "lstm":
        model = lstmModel(
            input_dim=sensorNum * 3,
            hidden_dim=128,
            lstm_dim=128,
            output_dim=sensorNum * 3,
        )
        # model = lstmModel()
    elif modelStr == "gru":
        model = GRuModel(
            input_dim=sensorNum * 3,
            hidden_dim=128,
            gru_dim=128,
            output_dim=sensorNum * 3,
        )
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

    # weights_csv_path = 'truss17Phi.csv'
    # load_weights_from_csv(weights_csv_path, model.linear2)
    # weight_matrix = pd.read_csv(weights_csv_path, header=None).values
    # weight_matrix = torch.tensor(weight_matrix, dtype=torch.float).transpose(0,1)
    # model.linear2.weight.data=weight_matrix
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
            )
            # log("loss:" + str(loss1))
            # log("reg:" + str(loss2))
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
                "truss_outputs/" + modelStr + "_best_model" + str(num) + ".pth",
            )
            model = model.to(device)
            ResultString = "epoch:" + str(epoch + 1) + ", minimum loss:" + str(min_loss)

    log(ResultString)
    model.to("cpu")
    log(str(model.linear2.weight.data))

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()


if __name__ == "__main__":

    main(modelStr, num)
