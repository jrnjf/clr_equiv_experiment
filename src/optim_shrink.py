"""Diagnostic optimizers for the weight-decay profile experiment (reviewer §4).

The question is whether a decay whose strength does NOT track the learning rate
destroys training, independent of its strength. torch.optim.SGD cannot express
that because its `weight_decay` is always multiplied by the current lr. This
subclass keeps every other SGD behaviour identical and applies a constant
multiplicative shrinkage before the update, matching SGD's ordering:

    SGD:     p <- p - lr*(g + wd*p) = p*(1 - lr*wd) - lr*g
    here:    p <- p*(1 - c)          then  p <- p - lr*g
"""
from __future__ import annotations

import torch


class ConstantShrinkSGD(torch.optim.SGD):
    """SGD with LR-independent constant shrinkage c (weight_decay must be 0)."""

    def __init__(self, params, shrink: float = 1e-4, **kwargs):
        if kwargs.get("weight_decay", 0.0):
            raise ValueError("ConstantShrinkSGD needs weight_decay=0; "
                             "use `shrink` instead")
        super().__init__(params, **kwargs)
        self.shrink = float(shrink)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                p.mul_(1.0 - self.shrink)
        return super().step(closure)
