"""
Vill -- Pre-Training Loop
----------------------------
Implements the core training loop for pre-training Vill from scratch
using next-token prediction with mixed precision, gradient accumulation,
cosine learning rate scheduling, and checkpoint management.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vill.model.transformer import VillForCausalLM
from vill.model.config import VillConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters for pre-training."""
    # Optimization
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000

    # Batch
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    effective_batch_size: int = 0  # computed

    # Duration
    max_steps: int = 50000
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000

    # Precision
    use_amp: bool = True  # Automatic mixed precision
    dtype: str = "bfloat16"

    # Paths
    output_dir: str = "checkpoints"
    backup_dir: Optional[str] = None   # e.g. Google Drive path — auto-copies every checkpoint
    resume_from: Optional[str] = None

    def __post_init__(self):
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps


def get_cosine_schedule(step: int, config: TrainingConfig) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step >= config.max_steps:
        return config.min_learning_rate
    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)


class Trainer:
    """
    Pre-training engine for Vill.

    Handles the complete training lifecycle: optimizer setup, learning
    rate scheduling, gradient accumulation, mixed precision, logging,
    and checkpoint save/restore.
    """

    def __init__(
        self,
        model: VillForCausalLM,
        train_loader: DataLoader,
        config: TrainingConfig,
        eval_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        self.model.to(self.device)

        # Optimizer: AdamW with weight decay applied only to 2D parameters
        decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
        nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            fused=torch.cuda.is_available(),
        )

        # Mixed precision -- only use on CUDA where it actually saves memory
        self.scaler = None
        self.amp_dtype = torch.float32
        self.use_amp = False
        if config.use_amp and self.device.type == "cuda":
            self.use_amp = True
            self.amp_dtype = torch.bfloat16
            if config.dtype == "float16":
                self.amp_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler()

        self.global_step = 0
        self.tokens_processed = 0

    def train(self) -> None:
        """Run the full pre-training loop."""
        logger.info(
            "Starting pre-training: %d params, device=%s, batch=%d, accum=%d, steps=%d",
            self.model.count_parameters(),
            self.device,
            self.config.batch_size,
            self.config.gradient_accumulation_steps,
            self.config.max_steps,
        )

        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)

        self.model.train()
        data_iter = iter(self.train_loader)
        start_time = time.time()
        running_loss = 0.0

        while self.global_step < self.config.max_steps:
            # Update learning rate
            lr = get_cosine_schedule(self.global_step, self.config)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # Accumulate gradients
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                self.tokens_processed += input_ids.numel()

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs["loss"] / self.config.gradient_accumulation_steps

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_loss += loss.item()

            # Gradient clipping
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.global_step += 1
            running_loss += accum_loss

            # Logging
            if self.global_step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                avg_loss = running_loss / self.config.log_interval
                tok_per_sec = self.tokens_processed / elapsed
                logger.info(
                    "step=%d  loss=%.4f  lr=%.2e  tok/s=%.0f  tokens=%d",
                    self.global_step, avg_loss, lr, tok_per_sec, self.tokens_processed,
                )
                running_loss = 0.0

            # Evaluation
            if self.eval_loader and self.global_step % self.config.eval_interval == 0:
                eval_loss = self._evaluate()
                logger.info("step=%d  eval_loss=%.4f", self.global_step, eval_loss)
                self.model.train()

            # Save checkpoint
            if self.global_step % self.config.save_interval == 0:
                self._save_checkpoint()

        self._save_checkpoint()
        logger.info("Pre-training complete. Total tokens: %d", self.tokens_processed)

    @torch.no_grad()
    def _evaluate(self, max_batches: int = 50) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        for batch in self.eval_loader:
            if count >= max_batches:
                break
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(input_ids=input_ids, labels=labels)
            total_loss += outputs["loss"].item()
            count += 1
        return total_loss / max(count, 1)

    def _save_checkpoint(self) -> None:
        import shutil
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"vill_step_{self.global_step}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "tokens_processed": self.tokens_processed,
            "config": self.model.config.__dict__,
        }, path)
        logger.info("Checkpoint saved: %s", path)

        # Auto-backup to Drive (or any second directory) if configured
        if self.config.backup_dir:
            backup_out = Path(self.config.backup_dir)
            backup_out.mkdir(parents=True, exist_ok=True)
            backup_path = backup_out / path.name
            # Skip if source and destination are the same file
            if path.resolve() != backup_path.resolve():
                shutil.copy2(path, backup_path)
                logger.info("Checkpoint backed up to Drive: %s", backup_path)
            else:
                logger.info("Checkpoint already in Drive (same path): %s", backup_path)

    def _load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.tokens_processed = checkpoint.get("tokens_processed", 0)
        logger.info("Resumed from step %d", self.global_step)
