import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# Define the transformer-based large language model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=2048, n_heads=24, num_layers=24, ff_hidden=8192, max_seq_length=1024):
        super(TransformerModel, self).__init__()

        self.d_model = d_model
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_length, d_model)

        # Multi-layer stack of decoder blocks
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, ff_hidden) for _ in range(num_layers)
        ])

        # Final linear layer for predicting the next word
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Memory or cache for chat context
        self.memory = []
        self.max_seq_length = max_seq_length

    def forward(self, input_ids, past_key_values=None):
        batch_size, seq_length = input_ids.size()
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(
            batch_size, seq_length)

        token_embeddings = self.embed_tokens(input_ids) + self.position_embeddings(positions)

        hidden_states = token_embeddings
        present_key_values = []

        # Loop over the layers and apply multi-head attention and feedforward layers
        for layer in self.layers:
            hidden_states, past_key_value = layer(hidden_states, past_key_values=past_key_values)
            present_key_values.append(past_key_value)

        logits = self.lm_head(hidden_states)
        return logits, present_key_values


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_hidden):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, ff_hidden)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, past_key_values=None):
        # Apply self-attention and add residual connection
        attention_output, past_key_value = self.self_attention(x, past_key_values=past_key_values)
        x = self.layer_norm1(x + attention_output)

        # Apply feed-forward network and add residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)

        return x, past_key_value


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, past_key_values=None):
        batch_size, seq_length, _ = x.size()

        # Linear transformations
        queries = self.query(x).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        keys = self.key(x).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        values = self.value(x).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, values)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.n_heads * self.d_k)
        output = self.out(attn_output)

        return output, (queries, keys, values)


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Example usage:
vocab_size = 50000
model = TransformerModel(vocab_size)
input_ids = torch.randint(0, vocab_size, (8, 1024))  # batch_size x seq_length
logits, _ = model(input_ids)
print(logits.shape)  # Output should be [8, 1024, vocab_size]
