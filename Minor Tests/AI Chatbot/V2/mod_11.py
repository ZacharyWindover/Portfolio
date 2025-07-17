import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Positional Embedding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# Multi-Head Attention with scaled dot-product
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, self.num_heads, 3 * self.d_k).transpose(1, 2)
        query, key, value = qkv.chunk(3, dim=-1)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        output = torch.matmul(attn, value)

        output = output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, d_model)
        return self.out_proj(output)


# Decoder Block for the transformer
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim):
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        return self.norm2(x + self.dropout(ff_output))


# Decoder-Only Transformer
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, ff_hidden_dim, max_seq_len=1024):
        super(DecoderOnlyTransformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, ff_hidden_dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.lm_head(x)


# Model Parameters
vocab_size = 50000
d_model = 2048
num_layers = 24
num_heads = 24
ff_hidden_dim = 8192
max_seq_len = 1024

# Initialize the model
model = DecoderOnlyTransformer(vocab_size, d_model, num_layers, num_heads, ff_hidden_dim, max_seq_len).to(device)
