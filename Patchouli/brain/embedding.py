
import torch.nn as nn

class RoleEmbedding(nn.Module):
    def __init__(self, vocab: int, d_model: int) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab, d_model)
        self.role_embed  = nn.Embedding(3, d_model)  # 0=sys 1=user 2=帕秋莉
        
    def forward(self, idx, role_ids=None):
        x = self.token_embed(idx)
        if role_ids is not None:
            x = x + self.role_embed(role_ids)
        return x