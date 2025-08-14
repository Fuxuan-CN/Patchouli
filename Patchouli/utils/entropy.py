
import re
import math
from collections import Counter

def information_entropy(text: str) -> float:
    """ 计算信息熵 """
    # 简单按字符级算熵，也可换成词级/BPE
    tokens = re.findall(r'\w|[\u4e00-\u9fff]', text.lower())
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    total = len(tokens)
    return -sum((c/total) * math.log2(c/total) for c in freq.values())