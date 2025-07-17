import torch
import torch.nn as nn
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = self.create_positional_encoding(max_len, d_model)

    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model).to(device)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((i + 1) / d_model)))
        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, encoded_words):
        embedding = self.embed(encoded_words) * math.sqrt(self.d_model)
        embedding += self.pe[:, :embedding.size(1)]
        return embedding


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super(MultiHeadAttention, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, mask):
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        query = query.view(query.size(0), -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.size(0), -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.size(0), -1, self.heads, self.d_k).permute(0, 2, 1, 3)

        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, value)

        context = context.permute(0, 2, 1, 3).contiguous().view(query.size(0), -1, self.heads * self.d_k)
        return self.concat(context)


class FeedForward(nn.Module):
    def __init__(self, d_model, middle_dim=2048):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, middle_dim)
        self.fc2 = nn.Linear(middle_dim, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, mask):
        attn_output = self.self_multihead(x, x, x, mask)
        x = self.layernorm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        return self.layernorm2(x + self.dropout2(ff_output))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.src_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_multihead(x, x, x, tgt_mask)
        x = self.layernorm1(x + self.dropout1(attn_output))
        enc_attn_output = self.src_multihead(x, enc_output, enc_output, src_mask)
        x = self.layernorm2(x + self.dropout2(enc_attn_output))
        ff_output = self.feed_forward(x)
        return self.layernorm3(x + self.dropout3(ff_output))


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, heads, max_len):
        super(Transformer, self).__init__()
        self.embed = Embeddings(vocab_size, d_model, max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask, tgt, tgt_mask):
        enc_output = self.encode(src, src_mask)
        return self.decode(tgt, enc_output, src_mask, tgt_mask)

    def encode(self, src, src_mask):
        x = self.embed(src)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.embed(tgt)
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.fc_out(x)



'''
Example usage
vocab_size = size of vocabulary
d_model = embedding size
num_layers = number of layers
heads = number of heads
max_len = max number of tokens model can process in a single input sequence
'''
vocab_size = 50000
d_model = 1024
num_layers = 24
heads = 24
max_len = 2048

transformer_model = Transformer(vocab_size, d_model, num_layers, heads, max_len).to(device)

# Assuming src and tgt are tokenized input sequences with appropriate masks
# src_mask = <mask for source>
# tgt_mask = <mask for target>
# output = transformer_model(src, src_mask, tgt, tgt_mask)
