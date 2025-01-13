import torch.nn as nn
import torch


class MrTeddyTD(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, seq_length):
        super(MrTeddyTD, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, embedding_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        transformer_out = self.transformer(embedded.transpose(0, 1)).transpose(0, 1)
        transformer_out = self.dropout(transformer_out)
        output = self.fc(transformer_out[:, -1, :])
        return output
