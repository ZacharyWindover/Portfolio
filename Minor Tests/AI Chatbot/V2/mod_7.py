import torch
import torch.nn as nn
import math


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

        # Linear projections and reshape for multi-head
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.fc_out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, hidden_dim, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.layernorm1(x + attn_output)
        ff_output = self.ff(x)
        return self.layernorm2(x + ff_output)


class GPTDecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, hidden_dim, max_seq_len, memory_len=1024,
                 dropout=0.1):
        super(GPTDecoderOnly, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.memory_len = memory_len

        # Token and positional embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.dropout = nn.Dropout(dropout)

        # Stack of decoder blocks
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)])

        # Final output layer
        self.layernorm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, past_memory=None, mask=None):
        batch_size, seq_len = x.shape
        device = x.device

        # Embedding + positional encoding
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :].to(device)
        x = self.dropout(x)

        # Concatenate memory with current input
        if past_memory is not None:
            x = torch.cat([past_memory, x], dim=1)

        # Truncate if necessary to manage memory size
        if x.shape[1] > self.memory_len:
            x = x[:, -self.memory_len:, :]

        # Attention mask for auto-regressive generation (future tokens masked out)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).unsqueeze(0)
        if mask is not None:
            causal_mask = mask & causal_mask

        # Pass through all decoder layers
        for layer in self.layers:
            x = layer(x, mask=causal_mask)

        # Normalize and project to vocab size
        x = self.layernorm(x)
        logits = self.fc_out(x)

        # Return the logits and optionally the memory
        return logits, x


# Example usage for setting up the model
vocab_size = 50000  # Adjust based on tokenizer
d_model = 2048  # Embedding size
num_layers = 24  # Number of decoder blocks
num_heads = 16  # Number of attention heads
hidden_dim = 8192  # Feedforward hidden dimension
max_seq_len = 1024  # Max sequence length for inputs
memory_len = 1024  # Memory length to store past tokens

model = GPTDecoderOnly(vocab_size, d_model, num_layers, num_heads, hidden_dim, max_seq_len, memory_len)
model.to('cuda' if torch.cuda.is_available() else 'cpu')
