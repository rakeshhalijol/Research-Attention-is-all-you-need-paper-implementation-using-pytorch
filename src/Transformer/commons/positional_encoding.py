import torch
import math
import torch.nn as nn
from typing_extensions import Annotated
from Transformer.abstract import Runnable


class PositionalEncoding(nn.Module, Runnable):
    def __init__(self,
                 max_len: Annotated[int, "It means how many no of words are there in a sequence"],
                 d_model: Annotated[int, "It tells in how many dimension each and every word represents"]) -> None:
        super().__init__()
        positional_encoder = torch.zeros(
            max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(
            1).float()  # (max_len, 1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-math.log(10000.0) / d_model))  # (d_model//2,)

        positional_encoder[:, 0::2] = torch.sin(position * div_term)
        positional_encoder[:, 1::2] = torch.cos(position * div_term)

        # Shape it to (1, max_len, d_model) for broadcasting with input: (batch_size, seq_len, d_model)
        positional_encoder = positional_encoder.unsqueeze(0)

        self.register_buffer("positional_encoder", positional_encoder)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        input_data: shape (batch_size, seq_len, d_model)
        returns: same shape with positional encoding added
        """
        seq_len = input_data.size(1)
        return input_data + self.positional_encoder[:, :seq_len]
