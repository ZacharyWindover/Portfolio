import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        # Linear transformations for query, key, and value
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # Output projection
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # Apply linear transformations and split into heads
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear projection and dropout
        return self.out(context)


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim=8192, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        return self.fc2(self.dropout(x))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, hidden_dim)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # Self-attention + skip connection
        attn_out = self.self_attn(x, mask)
        x = self.layernorm1(x + self.dropout(attn_out))

        # Feedforward + skip connection
        ff_out = self.ff(x)
        x = self.layernorm2(x + self.dropout(ff_out))

        return x


class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=2048, num_layers=24, num_heads=24, hidden_dim=8192, max_seq_len=1024):
        super(GPTModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_seq_len, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, hidden_dim) for _ in range(num_layers)])
        self.layernorm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def create_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * i / d_model)))
        return pe.unsqueeze(0)  # Shape (1, max_seq_len, d_model)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        pos_encoding = self.positional_encoding[:, :seq_len, :].to(x.device)

        # Embedding + Positional Encoding
        x = self.embedding(x) + pos_encoding

        # Pass through all decoder layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final layer normalization and linear projection to vocab size
        x = self.layernorm(x)
        logits = self.fc_out(x)

        return logits


# Example usage:
vocab_size = 50000  # Set your vocabulary size
model = GPTModel(vocab_size=vocab_size, d_model=2048, num_layers=24, num_heads=24, hidden_dim=8192)
model.to("cuda" if torch.cuda.is_available() else "cpu")
