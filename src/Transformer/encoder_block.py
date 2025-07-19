import math
import torch
import torch.nn as nn

from src.Transformer.commons.mha import MultiHeadAttention
from src.Transformer.commons.layer_norm import LayerNormalization
from src.Transformer.commons.ffn import FeedForward
from src.Transformer.abstract import Runnable


class EncoderBlock(nn.Module, Runnable):
    def __init__(self, seq_len: int, embed_dim: int, hidden_dim: int, num_heads: int, dropout=0.1, bias: bool = False):
        super().__init__()
        self.mha = MultiHeadAttention(
            num_heads=num_heads, embed_dim=embed_dim)
        self.ln1 = LayerNormalization(embed_dim=embed_dim)
        self.ln2 = LayerNormalization(embed_dim=embed_dim)
        self.ffn = FeedForward(
            embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head attention + residual + norm
        x_mha = self.mha(x)
        x = self.ln1(x + self.dropout(x_mha))

        # Feedforward + residual + norm
        x_ffn = self.ffn(x)
        x = self.ln2(x + self.dropout(x_ffn))
        return x


class Encoder(nn.Module, Runnable):
    def __init__(self, Nx, embed_dim, seq_len, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(seq_len, embed_dim, ff_hidden_dim, num_heads, dropout)
            for _ in range(Nx)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
