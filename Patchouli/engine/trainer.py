
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional
from ..schemas.train_method import TrainMethod
from ..utils.logger import logger
from ..utils.exec_hook import set_exechook
from ..knowledge import CharDataset
from ..brain.tokenizer import PatchouliTokenizer as DefaultTokenizer
from ..schemas.type_hints import ValidatorProtocol, TokenizerProtocol
from pyfiglet import Figlet
from ..information import AUTHOR, DESC, PURPLE_TEXT

class Trainer:
    """ 模型训练器 """

    def __init__(self, 
        cfg: TrainMethod, 
        model: nn.Module, 
        tokenizer: TokenizerProtocol = DefaultTokenizer(), 
        brain_validator: Optional[ValidatorProtocol] = None,
        default_fatal_hook: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.GradScaler] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    ) -> None:
        self._print_info()
        if default_fatal_hook: # 如果开启了默认钩子，则会设置
            set_exechook()
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
        self.optimizer = optimizer
        """ 优化器 """
        self.scaler = scaler
        """ 定标器 """
        logger.debug(f"模型: {model.__class__.__name__} 载入设备")
        self.model = model.to(self.device)
        """ 训练的模型 """
        self.accum_steps = self.cfg.gradient_accumulation_steps
        """ 梯度累计步数 """
        self.scheduler = scheduler
        """ 调度器 """
        logger.debug("载入完成，开始训练阶段...")
        if not self.validator:
            logger.info("brain_validator为空, 没有模型验证器, 后续训练将除开验证流程...")
        if self.use_cuda:
            cuda_mem_used = torch.cuda.memory_allocated(self.device) / 1024**3
            logger.success(f"模型: {self.model.__class__.__name__} 载入完成，载入使用的显存: {cuda_mem_used} GB")

    # ---------------- 内部工具 ----------------
    def _resolve_device(self) -> str:
        """ 自动处理合适的设备 """
        if self.cfg.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.cfg.device == "cuda" and not torch.cuda.is_available():
            logger.warning("指定 CUDA 不可用，自动降级到 CPU | 可能发生潜在的性能下降...")
            return "cpu"

        return self.cfg.device
    
    def _print_info(self) -> None:
        banner = Figlet(font='small').renderText('Patchouli')
        print(f"{PURPLE_TEXT.format(banner=banner)}")
        print(PURPLE_TEXT.format(banner=f"{AUTHOR}: 亲爱的作者提醒你，{DESC}\n"))
    
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
    
    def _save_model_epoch(self, epoch: int) -> None:
        """ 保存训练模型 """
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None
        }
        epoch_path = Path(self.cfg.export_model_path).with_suffix(f".epoch{epoch}.pth")
        torch.save(ckpt, epoch_path)
        logger.success(f"💾 已保存完整 checkpoint {epoch_path.name}")

    def _save_model_final(self) -> None:
        """ 保存最终模型 """
        return torch.save(self.model.state_dict(), self.cfg.export_model_path)

    def _judgment_continue_train(self, 
        _continue: Optional[Path]
    ) -> None:
        """ 判断是否继续训练，如果是将会加载模型 """
        if _continue and _continue.is_file():
            logger.info(f"确认了继续训练, 将指定模型载入: {self.device}")
            ckpt = torch.load(_continue, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"]) if self.optimizer else None
            self.scaler.load_state_dict(ckpt["scaler_state"]) if self.scaler else None
            self.scheduler.load_state_dict(ckpt["scheduler"]) if self.scheduler else None
            start_epoch = ckpt["epoch"] + 1
            logger.success(f"🔁 载入完成, 上次训练轮数: {start_epoch - 1}")
    
    def _monitor_system_health(self) -> None:
        """ 监控系统健康 """

        if self.use_cuda:
            reserved  = torch.cuda.memory_reserved() / 1024**3
            if reserved > self.cfg.warning_cuda_mem_usage:
                logger.warning(f"显存使用超过警报阈值设置: {reserved}GB > {self.cfg.warning_cuda_mem_usage}GB")
                torch.cuda.empty_cache()

    

    # -------------- 主训练入口 --------------
    def start(self, continue_from: Optional[str | Path] = None) -> None:
        """ 训练 """
        
        if self.optimizer is None:
            logger.debug("没有传入优化器，使用默认优化器 ")
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=0.1,
            )

        if self.scaler is None:
            logger.debug("没有传入打标器，使用默认打标器 ")
            self.scaler = torch.GradScaler(enabled=self.use_cuda)

        _continue = continue_from if isinstance(continue_from, (Path, type(None))) else Path(continue_from)
        dataloader = self._build_dataloader()

        start_epoch = 0
        total_tokens = 0
        epochs = 0

        self._judgment_continue_train(_continue)

        effective_batch = self.cfg.batch_size * self.cfg.gradient_accumulation_steps
        logger.info(
            f"🚀 开始训练, 共 {self.cfg.epoch} epoch, "
            f"micro-batch={self.cfg.batch_size}, "
            f"grad-accum={self.cfg.gradient_accumulation_steps}, "
            f"effective-batch={effective_batch}"
        )

        self.model.train()

        try:
            for epoch in range(start_epoch, self.cfg.epoch):
                epoch_loss = 0.0
                epochs = epoch
                self.optimizer.zero_grad()
                for batch_idx, (x, role, y) in enumerate(dataloader):
                    x, role, y = x.to(self.device), role.to(self.device), y.to(self.device)

                    with torch.autocast(
                        enabled=self.device.startswith("cuda"),
                        device_type=self.device,
                    ):
                        logits, new_past, loss = self.model(x, role, y)
                        loss = loss / self.accum_steps   # 平均到累计步

                    self.scaler.scale(loss).backward()

                    # 每 accum_steps 或最后一个 batch 真正更新
                    if (batch_idx + 1) % self.accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        if self.scheduler is not None:
                            self.scheduler.step()

                    tokens_in_batch = y.numel()
                    epoch_loss += loss.item() * tokens_in_batch * self.accum_steps
                    total_tokens += tokens_in_batch

                avg = epoch_loss / max(total_tokens, 1)
                logger.info(f"epoch {epoch} | token avg loss: {avg:.4f}")

                self._validate(epoch)
                # ✅ 每 save_every 个 epoch 保存一次
                if (epoch + 1) % self.cfg.save_every == 0 or epoch == self.cfg.epoch - 1:
                    if self.cfg.save_model_training: # 如果设定了训练要保存模型，则会保存
                        self._save_model_epoch(epoch)

                if (epoch + 1) % self.cfg.sys_health_check_every == 0 or epoch == self.cfg.sys_health_check_every - 1:
                    self._monitor_system_health()

        except KeyboardInterrupt:
            logger.info("用户终止了训练程序...")
            if self.cfg.interrupt_save:
                logger.info("正在保存模型...")
                self._save_model_epoch(epochs)
        else:
            # 训练完成会进入此块保存模型
            logger.info("所有轮次训练完成, 导出最终权重...")
            self._save_model_final()
            logger.success(f"🎉 训练完成！最终权重已经导出到: {self.cfg.export_model_path}")