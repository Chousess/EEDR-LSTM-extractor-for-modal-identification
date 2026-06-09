"""
传感器数量消融实验 - truss算例
对 sensorNum 从 MAX_SENSOR_NUM 逐步降至 3，
每个 sensorNum 枚举从 train_truss.py 的 sensorIndex2（5 个传感器）中选取 sensorNum 列的所有组合 C(5, sensorNum)，
分别训练 LSTM 模型并保存结果。

输出目录结构：
  sensor_numerical_truss/
    outputs/sensorNum{n}/  — .pth 模型权重 + _output.csv 隐变量
    logs/sensorNum{n}/     — 每个 sensorNum 一个汇总日志
    plot/sensorNum{n}/     — 损失曲线图

文件命名规则（传感器编号为 CSV 列的 1-based 索引）：
  n{sensorNum}_s{a}_{b}_..._{z}

断点续跑：若对应 .pth 已存在则跳过该组合。
"""

import sys
import os
import itertools
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

from models.lstm_model import lstmModel
from sklearn.preprocessing import StandardScaler
from reg import non_gaussian_penalty, covariance_penalty

# ─────────────────── 全局配置 ───────────────────
FILE_STR       = r"D:\\user\\xxh\\研究or项目\\LSTM\\LSTM汇总\\代码\\实验matlab\\spring truss\\truss_17.csv"
# 与 train_truss.py 保持一致
SENSOR_INDEX2  = np.array([5, 7, 9, 12, 15]) - 1   # 0-based 列索引，遍历池
MAX_SENSOR_NUM = 4          # 起始传感器数量（= len(SENSOR_INDEX2)）
MIN_SENSOR_NUM = 3          # 终止传感器数量（含）
MY_EPOCH      = 1000
BATCH_SIZE    = 128
HIDDEN_DIM    = 128
LSTM_DIM      = 128

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
PLOTS_DIR   = os.path.join(BASE_DIR, "plot")
# ────────────────────────────────────────────────


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def combo_tag(sensor_num: int, combo: tuple) -> str:
    """生成组合的唯一文件名标签，传感器编号为 1-based。"""
    indices_str = "_".join(str(c + 1) for c in combo)
    return f"n{sensor_num}_s{indices_str}"


def make_paths(sensor_num: int, tag: str):
    """返回该组合对应的各文件路径。"""
    out_dir  = os.path.join(OUTPUTS_DIR, f"sensorNum{sensor_num}")
    log_dir  = os.path.join(LOGS_DIR,    f"sensorNum{sensor_num}")
    plot_dir = os.path.join(PLOTS_DIR,   f"sensorNum{sensor_num}")
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    return {
        "pth":      os.path.join(out_dir,  f"{tag}.pth"),
        "csv":      os.path.join(out_dir,  f"{tag}_output.csv"),
        "log":      os.path.join(log_dir,  f"{tag}_log.txt"),
        "fig_loss": os.path.join(plot_dir, f"{tag}_loss.png"),
    }


def make_logger(log_path: str):
    """追加模式日志，同时打印到控制台。"""
    def log(message: str) -> None:
        print(message)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    return log


def make_summary_logger(sensor_num: int):
    """汇总日志：每个 sensorNum 一个总览文件（仅记录每组合的最终结果）。"""
    log_dir  = os.path.join(LOGS_DIR, f"sensorNum{sensor_num}")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"sensorNum{sensor_num}_summary.txt")
    return make_logger(log_path)


def train_one_combo(
    combo: tuple,
    sensor_num: int,
    data_scaled: np.ndarray,
    device: torch.device,
    paths: dict,
    log,
) -> float:
    """训练单个传感器组合，返回最佳 loss。"""
    set_seed(0)
    input_dim = sensor_num * 3  # 数据水平复制 3 份

    # 选列 + 标准化 + 三倍拼接
    selected = data_scaled[:, list(combo)]
    scaled   = StandardScaler().fit_transform(selected)
    scaled3  = np.hstack([scaled] * 3)

    X = torch.tensor(scaled3, dtype=torch.float).to(device)
    loader = DataLoader(TensorDataset(X, X), batch_size=BATCH_SIZE, shuffle=False)

    model = lstmModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        lstm_dim=LSTM_DIM,
        output_dim=input_dim,
    ).to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    criterion = nn.MSELoss().to(device)

    losses   = []
    min_loss = float("inf")

    for epoch in range(MY_EPOCH):
        total = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = (
                criterion(outputs[2], targets)
                + non_gaussian_penalty(outputs[1]) * 0.1
                + covariance_penalty(outputs[1]) * 0.1
            )
            loss.backward()
            optimizer.step()
            total += loss.item()

        avg = total / len(loader)
        losses.append(avg)

        if avg < min_loss:
            min_loss = avg
            model.to("cpu")
            torch.save(model.state_dict(), paths["pth"])
            model.to(device)

    # 损失曲线
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    tag = os.path.splitext(os.path.basename(paths["pth"]))[0]
    plt.title(tag)
    plt.tight_layout()
    plt.savefig(paths["fig_loss"], dpi=120)
    plt.close()

    # 推理并保存隐变量
    model_eval = lstmModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        lstm_dim=LSTM_DIM,
        output_dim=input_dim,
    )
    model_eval.load_state_dict(torch.load(paths["pth"], map_location="cpu"))
    model_eval.eval()
    X_cpu = torch.tensor(scaled3, dtype=torch.float)
    with torch.no_grad():
        Q = model_eval(X_cpu)[1]
    pd.DataFrame(Q.numpy()).to_csv(paths["csv"], index=False, header=None)

    return min_loss


def run_sensor_num(sensor_num: int, df_values: np.ndarray, device: torch.device) -> None:
    summary_log = make_summary_logger(sensor_num)
    all_combos  = list(itertools.combinations(SENSOR_INDEX2, sensor_num))
    total       = len(all_combos)

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_log(f"\n{'='*60}")
    summary_log(f"运行时间: {run_time}")
    summary_log(f"sensorNum = {sensor_num}, 共 {total} 种组合")
    summary_log(f"{'='*60}")

    for idx, combo in enumerate(all_combos, 1):
        tag   = combo_tag(sensor_num, combo)
        paths = make_paths(sensor_num, tag)

        # 断点续跑：跳过已完成的组合
        if os.path.exists(paths["pth"]):
            summary_log(f"[{idx}/{total}] 跳过（已存在）: {tag}")
            continue

        # 每个组合使用独立日志文件，文件名含传感器数量和序列
        combo_log = make_logger(paths["log"])
        combo_log(f"运行时间: {run_time}")
        combo_log(f"=== {tag} ===")

        summary_log(f"[{idx}/{total}] 开始: {tag}")
        try:
            best_loss = train_one_combo(combo, sensor_num, df_values, device, paths, combo_log)
            summary_log(f"[{idx}/{total}] 完成: {tag}  best_loss={best_loss:.6f}")
            combo_log(f"best_loss={best_loss:.6f}")
        except Exception as e:
            summary_log(f"[{idx}/{total}] 错误: {tag}  {e}")
            combo_log(f"错误: {e}")


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(FILE_STR)
    assert df.shape[1] >= max(SENSOR_INDEX2) + 1, (
        f"CSV 列数 {df.shape[1]} 不足以覆盖 SENSOR_INDEX2 中的最大索引 {max(SENSOR_INDEX2)}"
    )
    df_values = df.values.astype(float)

    for sensor_num in range(MAX_SENSOR_NUM, MIN_SENSOR_NUM - 1, -1):
        print(f"\n{'#'*60}")
        print(f"# sensorNum = {sensor_num}")
        print(f"{'#'*60}")
        run_sensor_num(sensor_num, df_values, device)

    print("\n全部消融实验完成。")


if __name__ == "__main__":
    main()
