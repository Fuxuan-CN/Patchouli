""" 数据路径常量 """
from pathlib import Path

BASE_DATA_PATH = Path(__file__).parent.parent.parent / "data"
""" 数据集文件夹路径 """
TRAIN_DATA_PATH = BASE_DATA_PATH / "train"
""" 训练数据集路径 """
VAL_DATA_PATH = BASE_DATA_PATH / "val"
""" 验证数据集路径 """
