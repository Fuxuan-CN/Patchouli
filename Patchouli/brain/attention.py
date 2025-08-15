""" 注意力机制模块 """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Rotary(nn.Module):
    """
    RoPE (Rotary Position Embedding) 实现
    """
    def __init__(self, dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        t = torch.arange(max_len, dtype=torch.float)[:, None] * inv_freq
        # 缓存到 buffer，不参与梯度
        self.register_buffer("cos", torch.cos(t).repeat_interleave(2, dim=-1))
        self.register_buffer("sin", torch.sin(t).repeat_interleave(2, dim=-1))

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int):
        cos, sin = self.cos[:seq_len], self.sin[:seq_len] # type: ignore
        return x * cos + self.rotate_half(x) * sin


class GQA(nn.Module):
    """
    Grouped-Query Attention
    """
    def __init__(self,
        d_model: int,
        n_head: int,
        n_kv_head: int,
        max_len: int
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.d_k = d_model // n_head

        self.q = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.k = nn.Linear(d_model, n_kv_head * self.d_k, bias=False)
        self.v = nn.Linear(d_model, n_kv_head * self.d_k, bias=False)
        self.o = nn.Linear(n_head * self.d_k, d_model, bias=False)

        self.rope = Rotary(self.d_k, max_len)

    def forward(self, x, mask=None, past_kv=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_head, self.d_k).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_head, self.d_k).transpose(1, 2)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)

        if self.n_kv_head != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        q = self.rope(q, k.size(-2))
        k = self.rope(k, k.size(-2))

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min) # 修改此处避免溢出
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o(out), (k, v)