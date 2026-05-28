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

        # Linear layers for query, key, and value
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        # Linear projections and reshape for multi-head
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Attention output
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
        x = self.layernorm1(x + attn_output)  # Add & Norm
        ff_output = self.ff(x)
        x = self.layernorm2(x + ff_output)    # Add & Norm
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, hidden_dim, max_seq_len=1024, dropout=0.1):
        super(GPTModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_seq_len, d_model)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)])
        self.layernorm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * i / d_model)))
        return pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :].to(x.device)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.layernorm(x)
        logits = self.fc_out(x)
        return logits

    def extend_context(self, past_memory, current_input):
        """Extend context by appending previous messages."""
        extended_input = torch.cat((past_memory, current_input), dim=1)
        return extended_input

# Example Usage
vocab_size = 50000  # You can change this based on your dataset
d_model = 2048  # Example embedding size
num_layers = 24  # Example number of layers
num_heads = 24  # Example number of attention heads
hidden_dim = 8192  # Example hidden dimension

model = GPTModel(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, num_heads=num_heads, hidden_dim=hidden_dim)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

