import math
import torch
import torch.nn as nn
from src.Transformer.abstract import Runnable


class TokenEmbeddings(nn.Module, Runnable):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeddings(x) * math.sqrt(self.embed_dim)
