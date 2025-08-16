""" 类型注解文件 """

# Patchouli/schemas/type_hints.py

import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Protocol

class TokenizerProtocol(Protocol):
    """ 分词器协议 """
    def __call__(self, content: str) -> list[int]:
        """ 分词 """
        return self.encode(content)

    def encode(self, s: str) -> list[int]:
        """ 编码实现 """
        ...

    def decode(self, ids: list[int]) -> str:
        """ 解码的实现 """
        ...

class ValidatorProtocol(Protocol):
    """ 模型验证器协议 """
    def __call__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str,
        global_step: int,
    ) -> float:
        """返回验证集 loss 或其他指标"""
        ...