

import torch
import math
import torch.nn.functional as F
from .position_embedding import RotaryEmbedding
from typing import Optional


class Embedding(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1,
    ):

        super().__init__()

        self.dim_model = embedding_size
        self.weight = torch.nn.parameter.Parameter(
            torch.empty(vocab_size, embedding_size, dtype=dtype)
        )

    def forward(self, ids: torch.Tensor):
        """
        Args:
            ids (:obj:`torch.Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
        Return:
            :obj:`torch.Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """  # noqa: E501

        embeds = F.embedding(ids, self.weight) / math.sqrt(self.dim_model)
        return embeds

    def projection(self, x: torch.Tensor):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, vocab_output_size)``: The projection output.
        """  # noqa: E501
        logits = F.linear(x / math.sqrt(self.dim_model), self.weight)
        return logits


class EmbeddingExt(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1,
        distance_scale: int = 16,
    ):

        super().__init__()

        self.dim_model = embedding_size
        self.rotary_emb = RotaryEmbedding(
            dim=embedding_size, distance_scale=distance_scale, dtype=dtype
        )

        self.weight = torch.nn.parameter.Parameter(
            torch.empty(vocab_size, embedding_size, dtype=dtype),
        )

    def forward(self, ids: torch.Tensor, ids_sub: torch.Tensor):
        """
        Args:
            ids (:obj:`torch.Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens.
            ids (:obj:`torch.Tensor` of shape ``(batch_size)``): Subscript of input sequence tokens.
        Return:
            :obj:`torch.Tensor` of shape ``(batch_size, seq_len, embedding_size)``: The embedding output.
        """  # noqa: E501

        embeds = F.embedding(ids, self.weight) / math.sqrt(self.dim_model)
        return self.rotary_emb(embeds, ids_sub)

    def projection(self, x: torch.Tensor, ext_table: Optional[torch.Tensor] = None):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
            ext_table (:obj:`torch.Tensor` of shape ``(ext_table_size, dim_model)``): Ext vocab table.
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, vocab_size + ext_table_size)``: The projection output.
        """  # noqa: E501
        logits = F.linear(x / math.sqrt(self.dim_model), self.weight)
        if ext_table is not None:
            logits_ext = F.linear(x, ext_table)
            logits = torch.cat([logits, logits_ext], dim=-1)
        return logits
