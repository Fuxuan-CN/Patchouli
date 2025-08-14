""" 知识库 """

from .path_location import TRAIN_DATA_PATH, VAL_DATA_PATH
from .dataset import CharDataset

__all__ = ["TRAIN_DATA_PATH", "VAL_DATA_PATH", "CharDataset"]