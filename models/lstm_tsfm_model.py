import torch
import torch.nn as nn

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_encoder_layers, num_decoder_layers):
        super(TransformerAutoencoder, self).__init__()
        self.linear1=nn.Linear(input_dim,hidden_dim)

        # Transformer 编码器
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim
            ),
            num_layers=num_encoder_layers
        )

        # Transformer 解码器
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim
            ),
            num_layers=num_decoder_layers
        )

        # LSTM 层，位于 Transformer 编码器和解码器之间
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        # 线性调整层，将 LSTM 的输出调整为与 Transformer 解码器相同的维度
        self.decoder_to_linear = nn.Linear(hidden_dim, input_dim)

        # 重建层，将解码的信号恢复成混合信号
        self.linear2 = nn.Linear(input_dim, input_dim)

    def forward(self, src):
        outputs=[]
        src=self.linear1(src)
        outputs.append(src)
        src = src.unsqueeze(0)  # 增加批次维度
        
        # Transformer编码
        encoder_output = self.encoder(src)

        # LSTM处理
        lstm_output, _ = self.lstm(encoder_output)
        
        # Transformer解码
        decoder_output = self.decoder(lstm_output, encoder_output)
        decoder_output=self.decoder_to_linear(decoder_output.squeeze(0))
        outputs.append(decoder_output)
        # 重建原始信号
        reconstructed = self.linear2(decoder_output)
        outputs.append(reconstructed)
        
        return outputs

