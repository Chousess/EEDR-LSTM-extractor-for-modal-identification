import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super(LSTMAutoencoder, self).__init__()

        # 编码器: 使用两层双向LSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate,
        )

        # 解码器: 使用单向LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_dim * 2,  # 注意这里输入维度是双向输出的两倍
            hidden_size=8,
            batch_first=True,
        )

        # 重建层，将解码的信号恢复成混合信号
        self.linear2 = nn.Linear(8, input_dim)

    def forward(self, x):
        outputs=[]
        x = x.unsqueeze(0)  # 增加一个批次大小的维度，适配LSTM输入
        outputs.append(x)
        # 编码
        encoder_outputs, _ = self.encoder(x)  # 获得双向LSTM的输出

        # 解码
        # 从编码器获取的最后一层输出被用作解码器的输入
        decoded, _ = self.decoder(encoder_outputs)
        decoded=decoded.squeeze(0)
        outputs.append(decoded)
        # 重建原始信号
        reconstructed = self.linear2(decoded)  # 移除批次维度
        outputs.append(reconstructed)

        return outputs
