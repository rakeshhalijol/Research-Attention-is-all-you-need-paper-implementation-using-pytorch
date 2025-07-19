import math
import torch
import torch.nn as nn
from typing import List, Annotated, Union
from src.Transformer.abstract import Runnable


class MultiHeadAttention(nn.Module, Runnable):
    def __init__(self, num_heads: Annotated[int, "No of self attention needed"],
                 embed_dim: Annotated[int, "dimension of each word"],
                 bias: Annotated[bool, "Required bias during trining"] = False,) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim % num_heads != 0"
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = self.embed_dim // self.num_heads
        self.wq = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.output_projection = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self,
                q_input: Annotated[torch.Tensor, "batch of data from the input data"],
                k_input: Union[torch.Tensor, None] = None,
                v_input: Union[torch.Tensor, None] = None,
                mask: Annotated[bool, "normal MHA or masked MHA?"] = False) -> torch.Tensor:
        batch = q_input.size(0)

        # Below code make sures algorithm is self attention not cross attention
        if k_input is None:
            k_input = q_input
        if v_input is None:
            v_input = q_input

        q = self.wq(q_input)
        k = self.wk(k_input)
        v = self.wv(v_input)

        # Sequence length can be diffrent for input data and output data
        T_q, T_k = q_input.size(1), k_input.size(1)

        # Split the q, k, v(embed_dim) dimension as (num_head, embed_dim / num_head)
        q = q.reshape(batch, T_q, self.num_heads,
                      self.head_dim).transpose(1, 2)
        k = k.reshape(batch, T_k, self.num_heads,
                      self.head_dim).transpose(1, 2)
        v = v.reshape(batch, T_k, self.num_heads,
                      self.head_dim).transpose(1, 2)

        # Calculate Attention
        k_transpose = k.transpose(-2, -1)  # (b, k, d) (b, d, p) = (b, k, p)
        score = (q @ k_transpose) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T_q, T_k), diagonal=1).bool(
        ) if mask else torch.zeros(T_q, T_k).bool()

        # Anyhow broadcasting works no need of unsqeeze but its good practice to
        # avoid broadcasting in Attentions, but clearly this step is optional
        mask = mask.unsqueeze(0).unsqueeze(0)
        score = score.masked_fill(mask, float("-inf"))
        attention_score = torch.softmax(score, dim=-1)
        attention = attention_score @ v

        # concat output of all heads
        attention = attention.transpose(1, 2)

        # Attention should have sequence of length = output sequence length.
        attention = attention.reshape(batch, T_q, self.embed_dim)

        # Since they are simple concatination to acutally mix all heads details we need a linear layer

        mha_output = self.output_projection(attention)
        return mha_output
