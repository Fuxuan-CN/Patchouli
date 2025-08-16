""" 测试聊天模块 """

import torch
import torch.nn as nn
from pathlib import Path
from ..brain.model import PatchouliModel
from ..brain.tokenizer import PatchouliTokenizer
from ..schemas.constant import DEFAULT_SYMBOL_DICT

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TOKENIZER = PatchouliTokenizer()
_EOS_ID = DEFAULT_SYMBOL_DICT.get("<|end|>", 65539)

class Session:
    """ 聊天会话 """
    def __init__(self, system: str, model_path: str | Path) -> None:
        self.system = system
        self._model = PatchouliModel().to(_DEVICE)
        self._model.load_state_dict(torch.load("patchouli_base.pth", map_location=_DEVICE))
        self._model.eval()
        self.history_ids = _TOKENIZER.encode(f"<|system|>{system}\n")
        self.role_ids = [0] * len(self.history_ids)
        self.past_kv = None

    def reply(self, user: str, max_new=128, temp=0.7, top_k=30):
        user_ids = _TOKENIZER.encode(f"<|user|>{user}\n<|patchouli|>")
        self.history_ids += user_ids
        self.role_ids += [1] * (len(user_ids) - 1) + [2]

        idx = torch.tensor([self.history_ids], device=_DEVICE)
        role = torch.tensor([self.role_ids], device=_DEVICE)

        out, self.past_kv = self._model.generate(
            idx[:, -self._model.max_len:],
            role[:, -self._model.max_len:],
            max_new=max_new,
            past_key_values=self.past_kv,
            eos_id=_EOS_ID,
        )
        reply_ids = out[0, len(self.history_ids):].tolist()
        reply_text = _TOKENIZER.decode(reply_ids)
        self.history_ids += reply_ids
        self.role_ids += [2] * len(reply_ids)
        return reply_text

# ----------- 使用示例 -----------
if __name__ == "__main__":
    s = Session("你是帕秋莉·诺蕾姬，图书馆的魔女。", Path("patchouli_base.pth"))
    while True:
        user = input("你：")
        if user == "exit":
            break
        print("帕秋莉：", s.reply(user))