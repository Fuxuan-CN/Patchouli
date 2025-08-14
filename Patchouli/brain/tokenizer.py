""" 分词器模块 """
from functools import lru_cache
from ..schemas.constant import DEFAULT_SYMBOL_DICT

class PatchouliTokenizer:
    """ 分词器 """
    def __init__(self, special_dict: dict[str, int] = DEFAULT_SYMBOL_DICT):
        self._special = special_dict
        self._rev_special = {v: k for k, v in self._special.items()}

    @lru_cache(maxsize=1024)
    def encode(self, s: str) -> list[int]:
        """ 编码 """
        ids = []
        i = 0
        while i < len(s):
            # 先尝试匹配特殊标记
            for marker, tok_id in self._special.items():
                if s.startswith(marker, i):
                    ids.append(tok_id)
                    i += len(marker)
                    break
            else:
                # 普通字符 → 直接转 Unicode 码点
                ids.append(ord(s[i]))
                i += 1
        return ids
    
    def __call__(self, content: str) -> list[int]:
        """ 把字符串编码成数字列表 """
        return self.encode(content)

    def decode(self, ids: list[int]) -> str:
        """ 解码 """
        chars = []
        for tid in ids:
            if tid in self._rev_special:
                chars.append(self._rev_special[tid])
            elif tid < 65536:
                chars.append(chr(tid))
            else:
                chars.append("�")   # 占位符
        return "".join(chars)