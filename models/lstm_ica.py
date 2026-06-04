class RNNWithSelfAttentionAndICA(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNNWithSelfAttentionAndICA, self).__init__()
        self.rnn = RNNModel(input_dim, hidden_dim, output_dim, num_layers)
        self.self_attention = SelfAttention(output_dim)
        self.ica = ICALayer(output_dim)
        self.reconstruction = ReconstructionLayer(output_dim)
    
    def forward(self, x):
        rnn_out = self.rnn(x)
        attn_out = self.self_attention(rnn_out)
        ica_out = self.ica(attn_out)
        reconstructed = self.reconstruction(ica_out)
        return ica_out, reconstructed