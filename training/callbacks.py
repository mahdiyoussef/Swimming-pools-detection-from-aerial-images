"""
Training Callbacks for Swimming Pool Detection System.

Author: Swimming Pool Detection Team
Date: 2026-01-02
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback to prevent overfitting."""

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.001,
        monitor: str = "val/mAP50",
        mode: str = "max"
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.best_value: Optional[float] = None
        self.counter = 0
        self.stopped = False

    def __call__(self, metrics: Dict[str, float]) -> bool:
        current = metrics.get(self.monitor)
        if current is None:
            return False
        
        if self.best_value is None:
            self.best_value = current
            return False
        
        improved = (
            (self.mode == "max" and current > self.best_value + self.min_delta) or
            (self.mode == "min" and current < self.best_value - self.min_delta)
        )
        
        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                logger.info(f"Early stopping triggered after {self.counter} epochs")
                return True
        
        return False


class ModelCheckpoint:
    """Save model checkpoints during training."""

    def __init__(
        self,
        output_dir: str,
        save_period: int = 10,
        save_best: bool = True,
        monitor: str = "val/mAP50",
        mode: str = "max",
        max_checkpoints: int = 5
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_period = save_period
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.max_checkpoints = max_checkpoints
        self.best_value: Optional[float] = None
        self.checkpoints: List[Path] = []

    def __call__(
        self,
        model: Any,
        epoch: int,
        metrics: Dict[str, float]
    ) -> Optional[Path]:
        saved_path = None
        current = metrics.get(self.monitor)
        
        # Save periodic checkpoint
        if epoch % self.save_period == 0:
            saved_path = self._save_checkpoint(model, epoch, metrics)
        
        # Save best checkpoint
        if self.save_best and current is not None:
            improved = (
                self.best_value is None or
                (self.mode == "max" and current > self.best_value) or
                (self.mode == "min" and current < self.best_value)
            )
            
            if improved:
                self.best_value = current
                best_path = self.output_dir / "best.pt"
                model.save(str(best_path))
                logger.info(f"Saved best model: {best_path}")
        
        self._cleanup_old_checkpoints()
        return saved_path

    def _save_checkpoint(
        self,
        model: Any,
        epoch: int,
        metrics: Dict[str, float]
    ) -> Path:
        mAP = metrics.get("val/mAP50", 0.0)
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}_mAP_{mAP:.4f}.pt"
        model.save(str(checkpoint_path))
        self.checkpoints.append(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def _cleanup_old_checkpoints(self) -> None:
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()


class ReduceLROnPlateau:
    """Reduce learning rate when metric plateaus."""

    def __init__(
        self,
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-6,
        monitor: str = "val/mAP50",
        mode: str = "max"
    ) -> None:
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.monitor = monitor
        self.mode = mode
        self.best_value: Optional[float] = None
        self.counter = 0

    def __call__(
        self,
        optimizer: Any,
        metrics: Dict[str, float]
    ) -> bool:
        current = metrics.get(self.monitor)
        if current is None:
            return False
        
        if self.best_value is None:
            self.best_value = current
            return False
        
        improved = (
            (self.mode == "max" and current > self.best_value) or
            (self.mode == "min" and current < self.best_value)
        )
        
        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._reduce_lr(optimizer)
                self.counter = 0
                return True
        
        return False

    def _reduce_lr(self, optimizer: Any) -> None:
        for param_group in optimizer.param_groups:
            old_lr = param_group["lr"]
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group["lr"] = new_lr
            logger.info(f"Reduced learning rate: {old_lr:.6f} -> {new_lr:.6f}")


class MetricsLogger:
    """Log training metrics to file and TensorBoard."""

    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_tensorboard = use_tensorboard
        self.writer = None
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.log_dir))
            except ImportError:
                logger.warning("TensorBoard not available")

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, step)

    def close(self) -> None:
        if self.writer:
            self.writer.close()
