import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
from models.lstm_model import lstmModel
from models.transformer_model import TransformerModel
from models.GRU_model import GRuModel
from datetime import datetime

# 配置参数
fileStr = "D:\\user\\xxh\\研究or项目\\LSTM\\LSTM汇总\\代码\\实验matlab\\flat\\flat_with_sequence\\data\\flat_selected.csv"
model_path = "flat_outputs/gru_best_model17.pth"
input_dim = 17 + 2
hidden_dim = 64
lstm_dim = 64
output_dim = 19

# 日志文件
log_file = "flat_logs/inference_log.txt"

def log_message(message):
    """打印并记录消息到日志文件"""
    print(message)
    with open(log_file, "a", encoding='utf-8') as f:
        f.write(message + "\n")

def infer_with_timing():
    """加载模型并进行推理，记录时间"""
    
    # 初始化设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_message(f"\n{'='*60}")
    log_message(f"推理开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"使用设备: {device}")
    log_message(f"{'='*60}\n")
    
    try:
        # 1. 加载数据
        log_message("[1/5] 正在加载数据...")
        start_load_data = time.time()
        df = pd.read_csv(fileStr)
        selected_data = df.values
        log_message(f"    数据形状: {selected_data.shape}")
        
        # 数据预处理（与训练时相同）
        first_column = selected_data[:, 0:1]
        last_column = selected_data[:, -1:]
        data_extended = np.hstack([first_column, selected_data, last_column])
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_extended)
        
        load_data_time = time.time() - start_load_data
        log_message(f"    数据加载完成，耗时: {load_data_time:.4f}秒")
        log_message(f"    处理后数据形状: {data_scaled.shape}\n")
        
        # 2. 创建模型
        log_message("[2/5] 正在创建模型...")
        start_create_model = time.time()
        model = GRuModel(input_dim=input_dim)
        model = model.to(device)
        create_model_time = time.time() - start_create_model
        log_message(f"    模型创建完成，耗时: {create_model_time:.4f}秒")
        log_message(f"\n模型结构:\n{str(model)}\n")
        
        # 3. 加载预训练权重
        log_message("[3/5] 正在加载预训练模型...")
        start_load_weights = time.time()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        load_weights_time = time.time() - start_load_weights
        log_message(f"    模型加载完成，耗时: {load_weights_time:.4f}秒")
        log_message(f"    模型路径: {model_path}\n")
        
        # 4. 进行推理
        log_message("[4/5] 正在进行推理...")
        X = torch.tensor(data_scaled, dtype=torch.float).to(device)
        
        # 记录推理时间
        start_inference = time.time()
        with torch.no_grad():
            outputs = model(X)
            # outputs[0]: 原始输入
            # outputs[1]: 解码器输出 (x2)
            # outputs[2]: 线性层输出 (x3)
            Q = outputs[1]  # 或者使用 outputs[2]，取决于需要
        inference_time = time.time() - start_inference
        
        log_message(f"    推理完成，耗时: {inference_time:.4f}秒")
        log_message(f"    输出形状: {Q.shape}\n")
        
        # 5. 保存结果
        log_message("[5/5] 正在保存结果...")
        start_save = time.time()
        
        # 转移到CPU并转换为numpy
        Q_numpy = Q.cpu().numpy()
        output_df = pd.DataFrame(Q_numpy)
        output_path = "flat_outputs/gru_inference_output_current.csv"
        output_df.to_csv(output_path, index=False, header=None)
        
        save_time = time.time() - start_save
        log_message(f"    结果保存完成，耗时: {save_time:.4f}秒")
        log_message(f"    输出路径: {output_path}\n")
        
        # 总结
        total_time = load_data_time + create_model_time + load_weights_time + inference_time + save_time
        log_message(f"{'='*60}")
        log_message("时间统计:")
        log_message(f"  - 数据加载: {load_data_time:.4f}秒")
        log_message(f"  - 模型创建: {create_model_time:.4f}秒")
        log_message(f"  - 权重加载: {load_weights_time:.4f}秒")
        log_message(f"  - 推理计算: {inference_time:.4f}秒 ⏱️")
        log_message(f"  - 结果保存: {save_time:.4f}秒")
        log_message(f"  - 总耗时: {total_time:.4f}秒")
        log_message(f"{'='*60}\n")
        
        return True, Q_numpy, inference_time
        
    except Exception as e:
        error_msg = f"推理出错: {str(e)}"
        log_message(f"\n❌ 错误: {error_msg}\n")
        import traceback
        log_message(traceback.format_exc())
        return False, None, 0

if __name__ == "__main__":
    success, outputs, inference_time = infer_with_timing()
    
    if success:
        log_message(f"✓ 推理成功完成!")
        log_message(f"  核心推理耗时: {inference_time:.4f}秒")
    else:
        log_message(f"✗ 推理失败，请查看日志信息")
