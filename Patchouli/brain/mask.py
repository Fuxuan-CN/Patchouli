import torch

def get_causal_mask(seq_len: int) -> torch.Tensor:
    
    return torch.tril(torch.ones(seq_len, seq_len))