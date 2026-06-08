from __future__ import annotations

from torch.optim.lr_scheduler import CosineAnnealingLR


class CosineAnnealingLRScheduler(CosineAnnealingLR):
    """
    Thin wrapper so Hydra can pass `total_steps` instead of `T_max`.
    """

    def __init__(self, optimizer, total_steps: int, **kwargs):
        super().__init__(optimizer, T_max=int(total_steps), **kwargs)
