"""
损失函数消融实验 - spring算例
四种情况：
  case 1 - no_cov:      loss = loss1 + non_gaussian_penalty * 0.1
  case 2 - no_loss2:    loss = loss1
  case 3 - no_nongauss: loss = loss1 + covariance_penalty * 0.1
  case 4 - full:        loss = loss1 + non_gaussian_penalty * 0.1 + covariance_penalty * 0.1
"""

import sys
import os
from datetime import datetime

# 使得可以引用上层目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免弹窗
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

from models.lstm_model import lstmModel
from sklearn.preprocessing import StandardScaler
from reg import non_gaussian_penalty, covariance_penalty

# ───────────────── 全局配置 ─────────────────
fileStr = r"D:\\user\\xxh\\研究or项目\\LSTM\\LSTM汇总\\代码\\实验matlab\\spring truss\\spring_AIAA.csv"
modelStr = "lstm"
myEpoch  = 2000
input_dim  = 6
batch_size = 128

CASE_NAMES = {
    1: "no_cov",
    2: "no_loss2",
    3: "no_nongauss",
    4: "full",
}

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
PLOTS_DIR   = os.path.join(BASE_DIR, "plot")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)
# ─────────────────────────────────────────────


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def make_logger(log_path: str):
    def log(message: str) -> None:
        print(message)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    return log


def compute_loss2(case: int, hidden):
    """根据case编号计算正则化损失"""
    if case == 1:
        return non_gaussian_penalty(hidden) * 0.1
    elif case == 2:
        return torch.tensor(0.0, device=hidden.device)
    elif case == 3:
        return covariance_penalty(hidden) * 0.1
    elif case == 4:
        return non_gaussian_penalty(hidden) * 0.1 + covariance_penalty(hidden) * 0.1
    else:
        raise ValueError(f"未知case编号: {case}")


def main(case: int):
    case_name = CASE_NAMES[case]
    plot_dir  = os.path.join(PLOTS_DIR, case_name)
    os.makedirs(plot_dir, exist_ok=True)

    log_path  = os.path.join(LOGS_DIR,    f"lstm_{case_name}_log.txt")
    pth_path  = os.path.join(OUTPUTS_DIR, f"lstm_{case_name}.pth")
    csv_path  = os.path.join(OUTPUTS_DIR, f"lstm_{case_name}_output.csv")
    fig_loss  = os.path.join(plot_dir,    f"lstm_{case_name}_loss.png")
    fig_resp  = os.path.join(plot_dir,    f"lstm_{case_name}_response.png")

    log = make_logger(log_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(0)

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"\n{'='*60}")
    log(f"运行时间: {run_time}")
    log(f"=== Case {case}: {case_name} ===")
    log(f"Using device: {device}")
    log(f"Data file: {fileStr}")

    df = pd.read_csv(fileStr)
    data = df.values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    X = torch.tensor(data_scaled, dtype=torch.float).to(device)
    dataset     = TensorDataset(X, X)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model      = lstmModel(input_dim=input_dim, hidden_dim=64, lstm_dim=64, output_dim=input_dim).to(device)
    model_eval = lstmModel(input_dim=input_dim, hidden_dim=64, lstm_dim=64, output_dim=input_dim)

    log(str(model))

    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    criterion = nn.MSELoss().to(device)

    losses    = []
    min_loss  = float("inf")
    best_info = ""

    for epoch in range(myEpoch):
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss1 = criterion(outputs[2], targets)
            loss2 = compute_loss2(case, outputs[1])
            loss  = loss1 + loss2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        log(f"Epoch {epoch+1}/{myEpoch}, Loss: {avg_loss:.6f}")

        if avg_loss < min_loss:
            min_loss  = avg_loss
            best_info = f"epoch: {epoch+1}, minimum loss: {min_loss:.6f}"
            model.to("cpu")
            torch.save(model.state_dict(), pth_path)
            model.to(device)

    log(best_info)

    # 损失曲线
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss - Case {case}: {case_name}")
    plt.tight_layout()
    plt.savefig(fig_loss, dpi=150)
    plt.close()

    # 推理并保存隐变量
    model_eval.load_state_dict(torch.load(pth_path, map_location="cpu"))
    model_eval.eval()

    X2 = torch.tensor(data_scaled, dtype=torch.float)
    with torch.no_grad():
        out = model_eval(X2)
        Q   = out[1]

    Q_df = pd.DataFrame(Q.numpy())
    Q_df.to_csv(csv_path, index=False, header=None)
    log(f"隐变量已保存至: {csv_path}")

    # 响应曲线
    time = np.linspace(0, 1, len(Q_df))
    fig, axes = plt.subplots(Q_df.shape[1], 1, figsize=(10, 2 * Q_df.shape[1]))
    if Q_df.shape[1] == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(time, Q_df.iloc[:, i])
        ax.set_title(f"Freedom {i+1}")
        ax.set_xlabel("Time")
        ax.grid(True)
    plt.suptitle(f"Response - Case {case}: {case_name}")
    plt.tight_layout()
    plt.savefig(fig_resp, dpi=150)
    plt.close()

    log(f"=== Case {case} 完成 ===\n")
    log(f"linear2 weights:\n{model_eval.linear2.weight.data}")


if __name__ == "__main__":
    # 用法: python train_spring_ablation.py <case_num>
    # case_num: 1=no_cov, 2=no_loss2, 3=no_nongauss, 4=full
    if len(sys.argv) < 2:
        print("用法: python train_spring_ablation.py <case_num>")
        print("  1 - no_cov      (不含covariance_penalty)")
        print("  2 - no_loss2    (不含loss2全部)")
        print("  3 - no_nongauss (不含non_gaussian_penalty)")
        print("  4 - full        (包含全部loss)")
        sys.exit(1)

    case_num = int(sys.argv[1])
    if case_num not in CASE_NAMES:
        print(f"错误: case_num 必须为 1~4，收到 {case_num}")
        sys.exit(1)

    main(case_num)
