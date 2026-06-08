from __future__ import annotations

from typing import Any, Dict, List
import math


class TriStageLRScheduler:
    """
    Minimal tri-stage LR scheduler (warmup -> hold -> decay).

    Designed to be Hydra-friendly and based on SpecAugment paper by Google Brain.
    config structure: configs.lr_scheduler.{lr, init_lr_scale, final_lr_scale,
    phase_ratio, total_steps}.
    """

    def __init__(
        self,
        optimizer,
        lr: float,
        total_steps: int,
        init_lr_scale: float = 0.01,
        final_lr_scale: float = 0.01,
        phase_ratio: List[float] = [0.1, 0.4, 0.5],
        group_names: List[str] | None = None,
    ):
        self.optimizer = optimizer
        self.base_lr = float(lr)
        self.init_lr = self.base_lr * float(init_lr_scale)
        self.final_lr = self.base_lr * float(final_lr_scale)
        self.total_steps = int(total_steps)

        if group_names is None:
            self.group_indices = list(range(len(self.optimizer.param_groups)))
        else:
            names = set(group_names)
            self.group_indices = [
                idx
                for idx, group in enumerate(self.optimizer.param_groups)
                if group.get("name") in names
            ]

        if not self.group_indices:
            raise ValueError("No optimizer param groups matched for scheduling")

        # Grabs initial lrs directly from the optimizer's param groups
        self.group_base_lrs = [
            self.optimizer.param_groups[idx]["lr"] for idx in self.group_indices
        ]

        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0")

        if len(phase_ratio) != 3:
            raise ValueError("phase_ratio must have exactly 3 elements [warmup, hold, decay]")

        self.warmup_steps = int(round(self.total_steps * phase_ratio[0]))
        self.hold_steps = int(round(self.total_steps * phase_ratio[1]))
        self.decay_steps = max(0, self.total_steps - self.warmup_steps - self.hold_steps)

        self.last_step = 0
        self._set_lr(self.init_lr)

    def _set_lr(self, lr: float) -> None:
        scale = 1.0 if self.base_lr <= 0 else lr / self.base_lr
        for idx, base_lr in zip(self.group_indices, self.group_base_lrs):
            self.optimizer.param_groups[idx]["lr"] = base_lr * scale

    def get_last_lr(self) -> List[float]:
        return [self.optimizer.param_groups[idx]["lr"] for idx in self.group_indices]

    def step(self) -> float:
        self.last_step += 1

        if self.last_step <= self.warmup_steps and self.warmup_steps > 0:
            scale = self.last_step / float(self.warmup_steps)
            lr = self.init_lr + scale * (self.base_lr - self.init_lr)
        elif self.last_step <= self.warmup_steps + self.hold_steps:
            lr = self.base_lr
        else:
            if self.decay_steps <= 0:
                lr = self.final_lr
            else:
                decay_step = min(self.last_step - self.warmup_steps - self.hold_steps, self.decay_steps)
                # Cosine annealing from base_lr to final_lr
                progress = decay_step / float(self.decay_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.final_lr + (self.base_lr - self.final_lr) * cosine

        self._set_lr(lr)
        return lr

    def state_dict(self) -> Dict[str, Any]:
        return {
            "last_step": self.last_step,
            "base_lr": self.base_lr,
            "init_lr": self.init_lr,
            "final_lr": self.final_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "hold_steps": self.hold_steps,
            "decay_steps": self.decay_steps,
            "group_base_lrs": self.group_base_lrs,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Strict loading: Let it raise a KeyError if the state dict is corrupted
        self.last_step = int(state_dict["last_step"])
        self.base_lr = float(state_dict["base_lr"])
        self.init_lr = float(state_dict["init_lr"])
        self.final_lr = float(state_dict["final_lr"])
        self.total_steps = int(state_dict["total_steps"])
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.hold_steps = int(state_dict["hold_steps"])
        self.decay_steps = int(state_dict["decay_steps"])
        self.group_base_lrs = list(state_dict["group_base_lrs"])
        
        current_lr = self.optimizer.param_groups[self.group_indices[0]]["lr"]
        self._set_lr(float(current_lr))
