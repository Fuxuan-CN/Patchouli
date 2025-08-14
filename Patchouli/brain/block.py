
from .attention import GQA
from .ffn import SwiGLU
import torch.nn as nn
from .norm import RMSNorm

class Block(nn.Module):
    def __init__(self,
        d_model: int, 
        n_head: int, 
        n_kv_head: int, 
        d_ff: int, 
        max_len: int
    ) -> None:
        super().__init__()
        self.attn=GQA(d_model,n_head,n_kv_head, max_len)
        self.ff=SwiGLU(d_model,d_ff)
        self.norm1=RMSNorm(d_model)
        self.norm2=RMSNorm(d_model)
    
    def forward(self, x, mask=None, past_kv=None):
        x1, present_kv = self.attn(self.norm1(x), mask, past_kv)
        x = x + x1
        x = x + self.ff(self.norm2(x))
        return x, present_kv