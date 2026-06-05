import os
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from models.base_model1 import base_model
from models.lstm_model import lstmModel
from models.GRU_model import GRuModel
from models.transformer_model import TransformerModel

# ======== 配置区（只需修改这里） ========
DATA_FILE   = "D:\\user\\xxh\\研究or项目\\LSTM\\LSTM汇总\\代码\\实验matlab\\spring truss\\truss_17.csv"
MODEL_PTH   = "truss_outputs/transformer_best_model17.pth"
SENSOR_NUM  = 5              # 选取的传感器数量，与训练时保持一致
PLOT        = False          # 是否绘制各分量图
TIMING_LOG  = "truss_outputs/timing_log.csv"   # 所有模型计时结果汇总
# ========================================

# 传感器列索引（与训练时完全相同）
_SENSOR_INDEX = np.array([5, 7, 9, 12, 15]) - 1   # 0-based

# INPUT_DIM / MODEL_TYPE / OUTPUT_CSV 自动推断，无需手动填写
_INPUT_DIM   = SENSOR_NUM * 3
_KNOWN_TYPES = ["lstm", "transformer", "gru", "base"]


def _auto_model_type(pth_path: str) -> str:
    name = os.path.splitext(os.path.basename(pth_path))[0].lower()
    for t in _KNOWN_TYPES:
        if name.startswith(t):
            return t
    raise ValueError(f"无法从文件名识别模型类型: {pth_path}，请确认文件名以模型类型开头")


def _auto_output_csv(pth_path: str, model_type: str) -> str:
    import re
    dir_  = os.path.dirname(pth_path)
    stem  = os.path.splitext(os.path.basename(pth_path))[0]
    match = re.search(r'\d+$', stem)
    num   = match.group() if match else stem
    return os.path.join(dir_, f"{model_type}_infer_{num}.csv").replace("\\", "/")


def build_model(model_type: str, input_dim: int) -> torch.nn.Module:
    if model_type == "base":
        return base_model(input_dim=input_dim)
    elif model_type == "lstm":
        return lstmModel(input_dim=input_dim, hidden_dim=128, lstm_dim=128, output_dim=input_dim)
    elif model_type == "gru":
        return GRuModel(input_dim=input_dim, hidden_dim=128, gru_dim=128, output_dim=input_dim)
    elif model_type == "transformer":
        return TransformerModel(input_dim=input_dim)
    else:
        raise ValueError(f"未知模型类型: {model_type}")


def save_timing(log_path: str, model_type: str, pth_path: str, times: list[float]):
    avg_ms = np.mean(times) * 1000
    min_ms = np.min(times)  * 1000
    max_ms = np.max(times)  * 1000
    std_ms = np.std(times)  * 1000
    row = {
        "model_type": model_type,
        "pth_file":   os.path.basename(pth_path),
        "n_runs":     len(times),
        "avg_ms":     round(avg_ms, 3),
        "min_ms":     round(min_ms, 3),
        "max_ms":     round(max_ms, 3),
        "std_ms":     round(std_ms, 3),
    }
    write_header = not os.path.exists(log_path)
    pd.DataFrame([row]).to_csv(log_path, mode="a", index=False, header=write_header)
    print(f"计时结果已追加到: {log_path}")


def preprocess(data_file: str):
    """与训练时完全相同：选取指定列 → StandardScaler → hstack * 3。"""
    df = pd.read_csv(data_file)
    selected = df.iloc[:, _SENSOR_INDEX[:SENSOR_NUM]]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(selected.values)
    data_scaled = np.hstack([data_scaled] * 3)
    return data_scaled, scaler


def main():
    model_type = _auto_model_type(MODEL_PTH)
    output_csv = _auto_output_csv(MODEL_PTH, model_type)

    print(f"数据文件:  {DATA_FILE}")
    print(f"模型类型:  {model_type}")
    print(f"权重文件:  {MODEL_PTH}")
    print(f"有效输入维度: {_INPUT_DIM}  ({SENSOR_NUM} 传感器 × 3)")

    data_scaled, _ = preprocess(DATA_FILE)
    X = torch.tensor(data_scaled, dtype=torch.float)

    model = build_model(model_type, _INPUT_DIM)
    model.load_state_dict(torch.load(MODEL_PTH, map_location="cpu"))
    model.eval()
    print(f"已加载模型权重: {MODEL_PTH}")

    N_RUNS = 10
    times: list[float] = []
    with torch.no_grad():
        for i in range(N_RUNS):
            t0 = time.perf_counter()
            outputs = model(X)
            _ = outputs[1].numpy()
            times.append(time.perf_counter() - t0)
            print(f"  第 {i+1:2d} 次推理: {times[-1]*1000:.3f} ms")
        Q = outputs[1]

    print(f"平均推理时间: {np.mean(times)*1000:.3f} ms  (共 {N_RUNS} 次)")

    Q_np = Q.numpy()
    Q_df = pd.DataFrame(Q_np)
    Q_df.to_csv(output_csv, index=False, header=None)
    print(f"推理完成，结果已保存: {output_csv}  (shape: {Q_np.shape})")
    save_timing(TIMING_LOG, model_type, MODEL_PTH, times)

    if PLOT:
        t_axis = np.linspace(0, 1, len(Q_df))
        for i in range(Q_df.shape[1]):
            plt.figure(figsize=(10, 4))
            plt.plot(t_axis, Q_df.iloc[:, i])
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title(f"[{model_type}] Inferred Component {i + 1}")
            plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
