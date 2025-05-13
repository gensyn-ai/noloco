import math

import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmupLR(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Current step
        current_step = self.last_epoch
        if current_step < self.warmup_steps:
            lr_scale = (current_step + 1) / self.warmup_steps
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        else:
            # After warmup, return max_lr (or any other decaying behavior)
            return [base_lr for base_lr in self.base_lrs]


class WarmupLRBuilder:
    def __init__(
        self,
        warmup_steps: int,
    ):
        self.warmup_steps = warmup_steps

    def build(self, optimizer: torch.optim.Optimizer):
        return WarmupLR(
            optimizer=optimizer,
            warmup_steps=self.warmup_steps,
        )


class CosineWarmupLR(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        min_lr: float,
        max_steps: int,
        warmup_steps: int,
        last_epoch=-1,
    ):
        self.min_lr = min_lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Current step
        current_step = self.last_epoch
        if current_step < self.warmup_steps:
            lr_scale = (current_step + 1) / self.warmup_steps
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        elif current_step >= self.max_steps:
            return [self.min_lr for _ in self.base_lrs]
        else:
            decay_ratio = (current_step - self.warmup_steps) / (
                self.max_steps - self.warmup_steps
            )
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            return [
                self.min_lr + coeff * (base_lr - self.min_lr)
                for base_lr in self.base_lrs
            ]


class CosineWarmupLRBuilder:
    def __init__(
        self,
        min_lr: float,
        max_steps: int,
        warmup_steps: int,
    ):
        assert max_steps > warmup_steps
        self.min_lr = min_lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps

    def build(self, optimizer: torch.optim.Optimizer):
        return CosineWarmupLR(
            optimizer=optimizer,
            min_lr=self.min_lr,
            max_steps=self.max_steps,
            warmup_steps=self.warmup_steps,
        )
