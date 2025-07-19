import math
import torch
import torch.nn as nn
from src.Transformer.abstract import Runnable


class LayerNormalization(nn.Module, Runnable):
    def __init__(self, embed_dim: int, eps: float = 1e-9) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(embed_dim)).float()
        self.beta = nn.Parameter(torch.ones(embed_dim)).float()
        self.eps = eps

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        # Assume input dim(2, 3, 6)
        mean = torch.mean(input_data, dim=-1, keepdim=True)  # (2, 3, 1)
        std = torch.std(input_data, dim=-1, keepdim=True)  # (2, 3, 1)

        # to normalize (2, 3, 6) - (2, 3, 1) = (2, 3, 6) due to broadcasting
        normalized_input_data = (input_data - mean) / (std + self.eps)

        # some weights do not require normalized output so alpha learnable parameter is introduced
        return self.alpha * normalized_input_data + self.beta
