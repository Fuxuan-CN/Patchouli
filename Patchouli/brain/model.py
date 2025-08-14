
import torch
import torch.nn as nn
from .block import Block
from .embedding import RoleEmbedding
from .norm import RMSNorm
from .mask import get_causal_mask

class PatchouliModel(nn.Module):
    def __init__(
        self,
        vocab=65536 + 256,
        d_model=2048,
        n_layer=24,
        n_head=32,
        n_kv_head=8,
        d_ff=8192,
        max_len=8192,
    ):
        super().__init__()
        self.embed = RoleEmbedding(vocab, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, n_kv_head, d_ff, max_len) for _ in range(n_layer)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.embed.token_embed.weight
        self.max_len = max_len

    # -------------- forward 支持 KV-Cache --------------
    def forward(self, idx, role_ids=None, targets=None, past_key_values=None):
        B, T = idx.shape
        x = self.embed(idx, role_ids)

        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)
            offset = 0
            mask = get_causal_mask(T).to(idx.device)
        else:
            offset = past_key_values[0][0].shape[-2]  # 已缓存长度
            mask = None  # 新 token 只看自己

        new_past = []
        for layer, (blk, past_kv) in enumerate(zip(self.blocks, past_key_values)):
            x, present_kv = blk(x, mask=mask, past_kv=past_kv)
            new_past.append(present_kv)

        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            logits = logits[:, -targets.size(1):, :]
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1
            )
        return logits, new_past, loss

    # -------------- generate 支持 KV-Cache --------------
    @torch.no_grad()
    def generate(
        self,
        idx,
        role_ids=None,
        max_new=128,
        temp=0.7,
        top_k=30,
        past_key_values=None,
        eos_id=65539,
    ):
        bsz, seq_len = idx.shape
        assert bsz == 1, "批量生成暂未实现"

        generated = []
        for _ in range(max_new):
            logits, past_key_values, _ = self(
                idx, role_ids, past_key_values=past_key_values
            )
            logits = logits[:, -1, :] / temp
            if top_k:
                topv, _ = torch.topk(logits, top_k)
                logits[logits < topv[..., -1, None]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            if next_id.item() == eos_id:
                break
            idx = next_id
            role_ids = torch.full_like(next_id, 2)  # 帕秋莉角色
            generated.append(next_id.item())

        return torch.cat([idx[:, :-max_new], idx], dim=1) if max_new else idx, past_key_values