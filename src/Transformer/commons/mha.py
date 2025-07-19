import math
import torch
import torch.nn as nn
from typing import List, Annotated
from src.Transformer.abstract import Runnable


class MultiHeadAttention(nn.Module, Runnable):
    def __init__(self, num_heads: Annotated[int, "No of self attention needed"],
                 embed_dim: Annotated[int, "dimension of each word"],
                 seq_length: Annotated[int, "Length of sentence after padding"],
                 bias: Annotated[bool, "Required bias during trining"] = False,
                 mask: Annotated[bool, "normal MHA or masked MHA?"] = False) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim % num_heads != 0"
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.head_dim = self.embed_dim // self.num_heads
        self.wq = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.output_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.require_mask = mask
        print("All parameters are set for multihead attention")

    def forward(self, batched_input_data: Annotated[torch.Tensor, "batch of data from the input data"]) -> torch.Tensor:
        batch = batched_input_data.size(0)
        q = self.wq(batched_input_data)
        k = self.wk(batched_input_data)
        v = self.wv(batched_input_data)

        # Split the q, k, v(embed_dim) dimension as (num_head, embed_dim / num_head)
        q = q.reshape(batch, self.seq_length, self.num_heads,
                      self.head_dim).transpose(1, 2)
        k = k.reshape(batch, self.seq_length, self.num_heads,
                      self.head_dim).transpose(1, 2)
        v = v.reshape(batch, self.seq_length, self.num_heads,
                      self.head_dim).transpose(1, 2)

        # Calculate Attention
        k_transpose = k.transpose(-2, -1)
        score = (q @ k_transpose) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(self.seq_length, self.seq_length), diagonal=1).bool(
        ) if self.require_mask else torch.zeros(self.seq_length, self.seq_length).bool()

        # Anyhow broadcasting works no need of unsqeeze but its good practice to
        # avoid broadcasting in Attentions, but clearly this step is optional
        mask = mask.unsqueeze(0).unsqueeze(0)
        score = score.masked_fill(mask, float("-inf"))
        attention_score = torch.softmax(score, dim=-1)
        attention = attention_score @ v

        # concat output of all heads
        attention = attention.transpose(1, 2)
        attention = attention.reshape(batch, self.seq_length, self.embed_dim)

        # Since they are simple concatination to acutally mix all heads details we need a linear layer

        mha_output = self.output_projection(attention)
        return mha_output

    def __call__(self, batched_input_data: Annotated[torch.Tensor, "batch of data from the input data"]):
        return self.forward(batched_input_data=batched_input_data)
