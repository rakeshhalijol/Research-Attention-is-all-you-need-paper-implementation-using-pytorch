import math
import torch
import torch.nn as nn
from typing import List, Annotated
from src.Transformer.abstract import Runnable


class FeedForward(nn.Module, Runnable):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1,  bias: bool = True):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=bias)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        layer = self.w1(input_data)
        output = self.relu(layer)
        dropout = self.dropout(output)
        layer = self.w2(dropout)

        return layer
