import math
import torch
import torch.nn as nn

from src.Transformer.commons.mha import MultiHeadAttention
from src.Transformer.commons.layer_norm import LayerNormalization
from src.Transformer.commons.ffn import FeedForward
from src.Transformer.abstract import Runnable


class DecoderBlock(nn.Module, Runnable):
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int, dropout=0.1, bias: bool = False):
        super().__init__()
        self.mmha = MultiHeadAttention(
            num_heads=num_heads, embed_dim=embed_dim)
        self.cmha = MultiHeadAttention(
            num_heads=num_heads, embed_dim=embed_dim)
        self.ln1 = LayerNormalization(embed_dim=embed_dim)
        self.ln2 = LayerNormalization(embed_dim=embed_dim)
        self.ln3 = LayerNormalization(embed_dim=embed_dim)
        self.ffn = FeedForward(
            embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output) -> torch.Tensor:
        # Multi-head attention + residual + norm
        x_mha = self.mmha(x, mask=True)
        x = self.ln1(x + self.dropout(x_mha))

        # Cross_Attention
        x_cmha = self.cmha(q_input=x, k_input=encoder_output,
                           v_input=encoder_output)
        x = self.ln2(x + self.dropout(x_cmha))

        # Feedforward + residual + norm
        x_ffn = self.ffn(x)
        x = self.ln3(x + self.dropout(x_ffn))
        return x


class Decoder(nn.Module):
    def __init__(self, Nx, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, ff_hidden_dim, num_heads, dropout)
            for _ in range(Nx)
        ])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x
