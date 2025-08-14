""" 数据集定义（智能识别：文件 or 文件夹，均支持循环/单文件） """

import re
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable
from ..utils.logger import logger

ROLE_RE = re.compile(r'(<\|system\|>|<\|user\|>|<\|patchouli\|>)')

class CharDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: Callable[[str], list[int]], block=1024) -> None:
        self.block = block
        self.tokenizer = tokenizer
        self.root = Path(path)

        # ---------- 智能判断 ----------
        if self.root.is_dir():
            self.files = sorted(self.root.glob("*.txt"))
            if not self.files:
                raise FileNotFoundError(f"{self.root} 目录下没有找到任何 .txt 文件")
            logger.info(f"检测到文件夹，共 {len(self.files)} 个 txt 文件")
            self._mode = "folder"
        elif self.root.is_file():
            self.files = [self.root]
            logger.info(f"检测到单文件：{self.root.name}")
            self._mode = "file"
        else:
            raise FileNotFoundError(f"{self.root} 既不是文件也不是文件夹")

        # ---------- 预加载 ----------
        self.current_file_idx = 0
        self._load_file(self.current_file_idx)

    # ----------- 内部工具 -----------
    def _load_file(self, idx: int):
        """加载第 idx 个文件"""
        file_path = self.files[idx % len(self.files)]
        logger.debug(f"加载数据文件：{file_path}")
        with file_path.open("rt", encoding="utf-8") as f:
            raw = f.read()
        self.tokens, self.roles = self._annotate_roles(raw)

    def _annotate_roles(self, data: str) -> tuple[list[int], list[int]]:
        """原有逻辑，不改动"""
        tokens, roles, last_end, current_role = [], [], 0, 0
        for m in ROLE_RE.finditer(data):
            seg = data[last_end:m.start()]
            seg_tokens = self.tokenizer(seg)
            tokens.extend(seg_tokens)
            roles.extend([current_role] * len(seg_tokens))
            current_role = {"<|system|>": 0, "<|user|>": 1, "<|patchouli|>": 2}[m.group(0)]
            last_end = m.end()
        tail = data[last_end:]
        tail_tokens = self.tokenizer(tail)
        tokens.extend(tail_tokens)
        roles.extend([current_role] * len(tail_tokens))
        return tokens, roles

    # ----------- Dataset 接口 -----------
    def __len__(self):
        # 每个文件可切多少个 block（近似）
        return max(1, len(self.tokens) // self.block)

    def __getitem__(self, idx):
        start = idx * self.block
        end = start + self.block + 1
        if end > len(self.tokens):
            # 单文件模式：从头再来（循环）
            # 多文件模式：跳到下一文件
            if self._mode == "folder":
                self._load_file(self.current_file_idx + 1)
            else:
                self._load_file(0)          # 单文件从头循环
            start, end = 0, self.block + 1  # 重置到新文件开头

        x = torch.tensor(self.tokens[start:end-1], dtype=torch.long)
        y = torch.tensor(self.tokens[start+1:end], dtype=torch.long)
        role = torch.tensor(self.roles[start:end-1], dtype=torch.long)
        return x, role, y

__all__ = ["CharDataset"]