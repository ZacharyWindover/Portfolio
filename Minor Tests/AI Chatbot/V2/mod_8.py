import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class GPTWithMemory(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, hidden_dim, max_seq_len=2048, dropout=0.1):
        super(GPTWithMemory, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.create_positional_encoding(max_seq_len, d_model)

        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)])

        self.layernorm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Memory/attention biases to emphasize recency
        self.recent_bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def create_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * i / d_model)))
        return pe.unsqueeze(0)

    def forward(self, x, past_context=None, mask=None):
        seq_len = x.size(1)
        embedded = self.embedding(x) + self.positional_encoding[:, :seq_len, :].to(x.device)

        if past_context is not None:
            # Concatenate past conversations (memory) with the current input
            embedded = torch.cat([past_context, embedded], dim=1)

        for layer in self.layers:
            embedded = layer(embedded, mask)

        # Normalize and output
        embedded = self.layernorm(embedded)
        logits = self.fc_out(embedded)

        return logits


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

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.layernorm1(x + attn_output)  # Add & Norm
        ff_output = self.ff(x)
        x = self.layernorm2(x + ff_output)  # Add & Norm
        return x


# Example Usage
vocab_size = 50000  # Define based on tokenizer
d_model = 2048  # Dimensionality of embeddings
num_layers = 24  # Adjust for model size
num_heads = 24  # Multi-head attention heads
hidden_dim = 8192  # Hidden layer size in feedforward
model = GPTWithMemory(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, num_heads=num_heads,
                      hidden_dim=hidden_dim)

# Move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
