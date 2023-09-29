

import torch
import math
import torch.nn.functional as F


class Linear(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1,
        scale_before: bool = False,
    ):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.scale_before = scale_before

        self.weight = torch.nn.parameter.Parameter(torch.empty((dim_out, dim_in), dtype=dtype))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        if self.scale_before:
            x = x / math.sqrt(self.dim_in)
            x = F.linear(x, self.weight)
        else:
            x = F.linear(x, self.weight)
            x = x / math.sqrt(self.dim_in)
        return x
