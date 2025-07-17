import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.fc_out(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, hidden_dim, dropout)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.layernorm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.layernorm2(x + ff_output)
        return x

class GPTModelWithMemory(nn.Module):
    def __init__(self, vocab_size, d_model=2048, num_layers=24, num_heads=24, hidden_dim=8192, max_seq_len=1024, dropout=0.1):
        super(GPTModelWithMemory, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_seq_len, d_model)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)])
        self.layernorm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Memory buffer to simulate extended context
        self.memory_buffer = None
        self.memory_size = 512  # Store 512 tokens of history in memory

    def create_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * i / d_model)))
        return pe.unsqueeze(0)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :].to(x.device)

        # If memory buffer is not None, prepend memory to input
        if self.memory_buffer is not None:
            x = torch.cat([self.memory_buffer, x], dim=1)

        for layer in self.layers:
            x = layer(x, mask)
        x = self.layernorm(x)
        logits = self.fc_out(x)

        # Update memory buffer
        self.memory_buffer = x[:, -self.memory_size:, :].detach()  # Store only last memory_size tokens

        return logits

# Example Usage
vocab_size = 50000  # Adjust based on your tokenizer
model = GPTModelWithMemory(vocab_size=vocab_size, d_model=2048, num_layers=24, num_heads=24, hidden_dim=8192)
model.to('cuda' if torch.cuda.is_available() else 'cpu')
