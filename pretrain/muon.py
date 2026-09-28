"""Muon optimizer with an AdamW fallback for non-matrix parameters."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def use_muon_for_parameter(name: str, parameter: torch.nn.Parameter) -> bool:
    """Match the Qwen4-Exp optimizer split used for backbone pretraining."""
    adamw_markers = (
        "embed",
        "lm_head",
        ".mlp.gate.",
        "shared_expert_gate",
        "hyper_connection",
        ".indexer.",
    )
    return parameter.ndim >= 2 and not any(marker in name for marker in adamw_markers)


def _zeroth_power_newton_schulz(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    """Approximate the polar factor in bf16 using the standard quintic iteration."""
    x = matrix.bfloat16()
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.mT if transposed else x


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """Apply Muon to selected matrices and AdamW to all remaining parameters."""

    def __init__(
        self,
        muon_params: Iterable[torch.nn.Parameter],
        adamw_params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        momentum: float = 0.95,
        ns_steps: int = 5,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
    ) -> None:
        groups = [
            {
                "params": list(muon_params),
                "algorithm": "muon",
                "lr": lr,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "ns_steps": ns_steps,
            },
            {
                "params": list(adamw_params),
                "algorithm": "adamw",
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": adam_betas,
                "eps": adam_eps,
            },
        ]
        super().__init__(groups, {"lr": lr, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            if group["algorithm"] == "muon":
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: dict) -> None:
        momentum = group["momentum"]
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            if gradient.ndim < 2:
                raise RuntimeError(f"Muon received a non-matrix parameter with shape {gradient.shape}")
            state = self.state[parameter]
            buffer = state.setdefault("momentum_buffer", torch.zeros_like(gradient))
            buffer.lerp_(gradient, 1.0 - momentum)
            update = gradient.lerp(buffer, momentum)  # Nesterov momentum
            update = _zeroth_power_newton_schulz(update, group["ns_steps"])
            # Leading dimensions are independent batches (for example, the
            # expert dimension in grouped-mm weights).
            update *= math.sqrt(max(1.0, gradient.shape[-2] / gradient.shape[-1]))
            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
            parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])

    def _adamw_step(self, group: dict) -> None:
        beta1, beta2 = group["betas"]
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            state = self.state[parameter]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.lerp_(gradient, 1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            step = state["step"]
            step_size = group["lr"] * math.sqrt(1.0 - beta2**step) / (1.0 - beta1**step)
            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
            parameter.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group["eps"]), value=-step_size)
