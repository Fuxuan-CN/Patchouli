""" 训练方案配置模块 """

import attr
from typing import Literal
from pathlib import Path
from ..knowledge import TRAIN_DATA_PATH, VAL_DATA_PATH

@attr.s(frozen=True)
class TrainMethod:
    """ 训练方案 """
    device: Literal["cuda", "cpu", "auto"] = attr.ib("auto")
    """ 使用的设备 """
    epoch: int = attr.ib(5)
    """ 训练轮次 """
    batch_size: int = attr.ib(16)
    """ 一批次多少数据 """
    learning_rate: float = attr.ib(3e-4)
    """ 学习率 """
    block: int = attr.ib(1024)
    """ 块大小 """
    export_model_path: str | Path = attr.ib("")
    """ 模型导出文件路径 """
    dataset_path: str | Path = attr.ib(TRAIN_DATA_PATH)
    """ 数据集文件路径 """
    val_dataset_path: str | Path = attr.ib(VAL_DATA_PATH)
    """ 验证集文件路径 """
    loader_shuffle: bool = attr.ib(True)
    """ 是否打乱 """
    loader_num_workers: int = attr.ib(0)
    """ 数据加载器线程数 """
    save_every: int = attr.ib(100)
    """ 多少轮保存一次 """
    val_every: int = attr.ib(2)
    """ 多少轮验证一次 """
    warning_cuda_mem_usage: float = attr.ib(10)
    """ 显存警报阈值 """
    sys_health_check_every: int = attr.ib(2)
    """ 每多少次训练后, 检查一次系统健康 """
    interrupt_save: bool = attr.ib(False)
    """ 终止按下的时候是否保存模型 """

DEFAULT = TrainMethod(
    device="cuda",
    epoch=5,
    batch_size=2,           # RTX 2060 12G 建议别开太大
    learning_rate=3e-4,
    block=1024,
    dataset_path=TRAIN_DATA_PATH,
    val_dataset_path=VAL_DATA_PATH,
    export_model_path=Path("patchouli_base.pth"),
    loader_shuffle=True,
    loader_num_workers=0,
    save_every=100,
    val_every=10,
    warning_cuda_mem_usage=11,
    sys_health_check_every=2,
    interrupt_save=False
)
""" 预设训练方案 """