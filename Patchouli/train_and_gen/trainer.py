
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional
from ..schemas.train_method import TrainMethod
from ..utils.logger import logger
from ..knowledge import CharDataset
from ..brain.tokenizer import PatchouliTokenizer as DefaultTokenizer
from ..schemas.type_hints import ValidatorProtocol, TokenizerProtocol

class Trainer:
    """ 模型训练器 """

    def __init__(self, 
        cfg: TrainMethod, 
        model: nn.Module, 
        tokenizer: TokenizerProtocol = DefaultTokenizer(), 
        brain_validator: Optional[ValidatorProtocol] = None
    ) -> None:
        logger.debug(f"初始化训练器...")
        self.validator = brain_validator
        """ 验证器 """
        self.cfg = cfg
        """ 训练器配置 """
        self.use_cuda = self.cfg.device.startswith("cuda")
        """ 是否使用显卡 """
        self.device = self._resolve_device()
        """ 训练用的设备 """
        logger.debug(f"使用的设备: {self.device}")
        self.tokenizer = tokenizer
        """ 分词器 """
        logger.debug(f"模型: {model.__class__.__name__} 载入设备")
        self.model = model.to(self.device)
        """ 训练的模型 """
        logger.debug("载入完成，开始训练阶段...")
        if not self.validator:
            logger.info("brain_validator为空, 没有模型验证器, 后续训练将除开验证流程...")
        if self.use_cuda:
            cuda_mem_used = torch.cuda.memory_allocated(self.device) / 1024**3
            logger.success(f"模型: {self.model.__class__.__name__} 载入完成，载入使用的显存: {cuda_mem_used} GB")

    # ---------------- 内部工具 ----------------
    def _resolve_device(self) -> str:
        if self.cfg.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.cfg.device == "cuda" and not torch.cuda.is_available():
            logger.warning("指定 CUDA 不可用，自动降级到 CPU | 可能发生潜在的性能下降...")
            return "cpu"

        return self.cfg.device
    
    def _validate(self, epoch: int) -> None:
        """ 验证模型，如果有模型验证器的话 """
        if self.validator is not None and epoch % self.cfg.val_every == 0:
            logger.info("开始验证...")
            val_loss = self.validator(
                self.model, self._build_dataloader(val=True), self.device, epoch
            )
            logger.info(f"验证结果: epoch {epoch} | val_loss={val_loss:.4f}")

    def _build_dataloader(self, val: bool = False) -> DataLoader:
        """ 构建数据集加载器 """
        ds = CharDataset(self.cfg.dataset_path, tokenizer=self.tokenizer, block=self.cfg.block) \
        if not val else CharDataset(self.cfg.val_dataset_path, self.tokenizer, self.cfg.block)

        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.loader_shuffle,
            num_workers=self.cfg.loader_num_workers,
            pin_memory=self.use_cuda,
        )
    
    def _save_model_epoch(self, epoch: int, optimizer: torch.optim.Optimizer, scaler: torch.GradScaler) -> None:
        """ 保存训练模型 """
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            # 如果有 scheduler，也塞进来
        }
        epoch_path = Path(self.cfg.export_model_path).with_suffix(f".epoch{epoch}.pth")
        torch.save(ckpt, epoch_path)
        logger.success(f"💾 已保存完整 checkpoint {epoch_path.name}")

    def _save_model_final(self) -> None:
        """ 保存最终模型 """
        return torch.save(self.model.state_dict(), self.cfg.export_model_path)

    def _judgment_continue_train(self, _continue: Optional[Path], optimizer: torch.optim.Optimizer, scaler: torch.GradScaler) -> None:
        """ 判断是否继续训练，如果是将会加载模型 """
        if _continue and _continue.is_file():
            logger.info(f"确认了继续训练, 将指定模型载入: {self.device}")
            ckpt = torch.load(_continue, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = ckpt["epoch"] + 1
            logger.success(f"🔁 载入完成, 上次训练轮数: {start_epoch - 1}")
    
    def _monitor_system_health(self) -> None:
        """ 监控系统健康 """

        if self.use_cuda:
            reserved  = torch.cuda.memory_reserved() / 1024**3
            if reserved > self.cfg.warning_cuda_mem_usage:
                logger.warning(f"显存使用超过警报阈值设置: {reserved}GB > {self.cfg.warning_cuda_mem_usage}GB")

    # -------------- 主训练入口 --------------
    def start(self, continue_from: Optional[str | Path] = None) -> None:
        """ 训练 """
        _continue = continue_from if isinstance(continue_from, (Path, type(None))) else Path(continue_from)
        dataloader = self._build_dataloader()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=0.1,
        )
        scaler = torch.GradScaler(enabled=self.use_cuda)

        start_epoch = 0
        total_tokens = 0
        epochs = 0

        self._judgment_continue_train(_continue, optimizer, scaler)

        logger.info(f"🚀 开始训练, 训练{self.cfg.epoch}轮, 每轮{self.cfg.batch_size}批数据")

        self.model.train()

        try:
            for epoch in range(start_epoch, self.cfg.epoch):
                epoch_loss = 0.0
                epochs = epoch
                for x, role, y in dataloader:
                    x, role, y = x.to(self.device), role.to(self.device), y.to(self.device)

                    with torch.autocast(
                        enabled=self.device.startswith("cuda"),
                        device_type=self.device,
                    ):
                        _, loss = self.model(x, role, y)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    tokens_in_batch = y.numel()
                    epoch_loss += loss.item() * tokens_in_batch
                    total_tokens += tokens_in_batch

                avg = epoch_loss / max(total_tokens, 1)
                logger.info(f"epoch {epoch} | token avg loss: {avg:.4f}")

                self._validate(epoch)
                # ✅ 每 save_every 个 epoch 保存一次
                if (epoch + 1) % self.cfg.save_every == 0 or epoch == self.cfg.epoch - 1:
                    self._save_model_epoch(epoch, optimizer, scaler)

                if (epoch + 1) % self.cfg.sys_health_check_every == 0 or epoch == self.cfg.sys_health_check_every - 1:
                    self._monitor_system_health()

        except KeyboardInterrupt:
            logger.info("用户终止了训练程序...")
            if self.cfg.interrupt_save:
                logger.info("正在保存模型...")
                self._save_model_epoch(epochs, optimizer, scaler)
        else:
            # 训练完成会进入此块保存模型
            logger.info("所有轮次训练完成, 导出最终权重...")
            self._save_model_final()
            logger.success(f"🎉 训练完成！最终权重已经导出到: {self.cfg.export_model_path}")