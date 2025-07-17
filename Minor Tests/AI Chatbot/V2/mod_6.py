import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.embed_size = embed_size

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.embed_size // self.heads)
        keys = keys.reshape(N, key_len, self.heads, self.embed_size // self.heads)
        queries = query.reshape(N, query_len, self.heads, self.embed_size // self.heads)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )
        out = self.fc_out(out)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, key, value, trg_mask)
        x = self.dropout(self.norm(attention + x))
        forward = self.feed_forward(x)
        x = self.dropout(self.norm(forward + x))
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        vocab_size,
        max_length,
        dropout
    ):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([embed_size])).to(device)

    def forward(self, x, trg_mask, memory=None):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, None, trg_mask)

        out = self.fc_out(out)
        return out

    def generate_memory(self, x, max_length):
        return self.forward(x, None)[:, -max_length:, :]

# Memory mechanism to retain and refer to previous conversations
class Memory:
    def __init__(self, max_memory_length=1024):
        self.memory = []
        self.max_memory_length = max_memory_length

    def update(self, new_data):
        self.memory.append(new_data)
        if len(self.memory) > self.max_memory_length:
            self.memory.pop(0)

    def retrieve_memory(self):
        return self.memory

# Example instantiation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_size = 2048
num_layers = 24
heads = 24
forward_expansion = 4
vocab_size = 50000
max_length = 1024
dropout = 0.1

model = Transformer(embed_size, num_layers, heads, forward_expansion, vocab_size, max_length, dropout).to(device)
