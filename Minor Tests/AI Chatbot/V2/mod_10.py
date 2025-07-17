import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTDecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_mult=4, dropout=0.1):
        super(GPTDecoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

        # Multi-head attention (decoder-only)
        self.attn = nn.MultiheadAttention(embed_size, heads, dropout=dropout)

        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_mult * embed_size),
            nn.GELU(),
            nn.Linear(ff_hidden_mult * embed_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None):
        # Multi-head self-attention
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = attn_output + x
        x = self.ln1(x)

        # Feedforward block
        ff_output = self.ff(x)
        x = ff_output + x
        x = self.ln2(x)

        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, max_seq_len, ff_hidden_mult=4, dropout=0.1):
        super(GPT, self).__init__()

        # Token embeddings and positional embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(max_seq_len, embed_size)
        self.dropout = nn.Dropout(dropout)

        # Stack of GPTDecoderBlocks (decoder-only layers)
        self.blocks = nn.ModuleList(
            [GPTDecoderBlock(embed_size, heads, ff_hidden_mult, dropout) for _ in range(num_layers)]
        )

        self.ln_f = nn.LayerNorm(embed_size)  # Final layer norm
        self.fc_out = nn.Linear(embed_size, vocab_size)  # Output layer to predict next token

    def forward(self, x, attn_mask=None):
        batch_size, seq_len = x.size()

        # Token + positional embeddings
        positions = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, seq_len).to(x.device)
        x = self.token_embed(x) + self.pos_embed(positions)
        x = self.dropout(x)

        # Pass through decoder blocks
        for block in self.blocks:
            x = block(x, attn_mask)

        # Final layer normalization and output
        x = self.ln_f(x)
        logits = self.fc_out(x)

        return logits
