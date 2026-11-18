"""Session-local fast weights.

Fast weights live in a :class:`FastMemoryState`, never in the global model
parameters.  Adapting a session therefore cannot mutate slow weights.  All
updates are functional: new tensors are produced rather than writing through
``.data`` on shared parameters.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _clip_tensor(g: torch.Tensor, clip: float) -> torch.Tensor:
    if clip <= 0:
        return g
    norm = g.norm()
    if not torch.isfinite(norm) or norm <= clip:
        return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return g * (clip / (norm + 1e-12))


@dataclass
class FastMemoryState:
    """Bounded, serialisable session memory."""

    kind: str
    params: dict[str, torch.Tensor]
    updates: int = 0
    adapt_seconds: float = 0.0
    last_loss: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "FastMemoryState":
        return FastMemoryState(
            kind=self.kind,
            params={k: v.detach().clone() for k, v in self.params.items()},
            updates=self.updates,
            adapt_seconds=self.adapt_seconds,
            last_loss=self.last_loss,
            metadata=dict(self.metadata),
        )

    def to(self, device: torch.device | str) -> "FastMemoryState":
        self.params = {k: v.to(device) for k, v in self.params.items()}
        return self

    def parameter_count(self) -> int:
        return int(sum(v.numel() for v in self.params.values()))

    def state_bytes(self) -> int:
        return int(
            sum(v.numel() * v.element_size() for v in self.params.values())
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "kind": self.kind,
                "params": {k: v.detach().cpu() for k, v in self.params.items()},
                "updates": self.updates,
                "adapt_seconds": self.adapt_seconds,
                "last_loss": self.last_loss,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "FastMemoryState":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            kind=payload["kind"],
            params=payload["params"],
            updates=int(payload.get("updates", 0)),
            adapt_seconds=float(payload.get("adapt_seconds", 0.0)),
            last_loss=payload.get("last_loss"),
            metadata=payload.get("metadata", {}),
        )


class TTTLinear(nn.Module):
    """A linear map adapted online with gradient descent on a self-supervised
    regression objective.

    Given key features ``x`` and target features ``y`` derived from the same
    input stream, the inner loop minimises ``MSE(Wx + b, y)``.  It is an
    experimental TTT-style mechanism, not a reproduction of any published TTT
    architecture.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        lr: float = 0.05,
        steps: int = 2,
        clip: float = 1.0,
        weight_decay: float = 0.0,
        decay: float = 0.0,
        init_scale: float = 0.02,
        inner_optimizer: str = "adam",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr
        self.steps = steps
        self.clip = clip
        self.weight_decay = weight_decay
        self.decay = decay
        self.inner_optimizer = inner_optimizer
        self.register_buffer(
            "init_weight", torch.randn(out_dim, in_dim) * init_scale
        )
        self.register_buffer("init_bias", torch.zeros(out_dim))

    def init_state(self, device: torch.device | str = "cpu") -> FastMemoryState:
        return FastMemoryState(
            kind="ttt_linear",
            params={
                "weight": self.init_weight.detach().clone().to(device),
                "bias": self.init_bias.detach().clone().to(device),
            },
        )

    @staticmethod
    def functional_forward(
        params: dict[str, torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        return F.linear(x, params["weight"], params["bias"])

    def read(self, state: FastMemoryState, queries: torch.Tensor) -> torch.Tensor:
        if state.kind != "ttt_linear":
            raise ValueError(f"state kind {state.kind!r} cannot be read by TTTLinear")
        return self.functional_forward(state.params, queries)

    def adapt(
        self,
        state: FastMemoryState,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> FastMemoryState:
        if state.kind != "ttt_linear":
            raise ValueError(f"state kind {state.kind!r} cannot be adapted by TTTLinear")
        if keys.ndim != 2 or values.ndim != 2:
            raise ValueError("keys and values must be [N, dim]")
        if keys.shape[0] != values.shape[0]:
            raise ValueError("keys and values must have the same number of rows")
        keys = torch.nan_to_num(keys.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        values = torch.nan_to_num(values.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        w = state.params["weight"].detach().clone().requires_grad_(True)
        b = state.params["bias"].detach().clone().requires_grad_(True)
        m_w = torch.zeros_like(w)
        v_w = torch.zeros_like(w)
        m_b = torch.zeros_like(b)
        v_b = torch.zeros_like(b)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        last_loss = 0.0
        for step_idx in range(1, max(1, self.steps) + 1):
            pred = F.linear(keys, w, b)
            loss = F.mse_loss(pred, values)
            gw, gb = torch.autograd.grad(loss, (w, b), allow_unused=False)
            gw = _clip_tensor(gw, self.clip)
            gb = _clip_tensor(gb, self.clip)
            with torch.no_grad():
                if self.weight_decay:
                    gw = gw + self.weight_decay * w
                if self.inner_optimizer == "adam":
                    m_w = beta1 * m_w + (1 - beta1) * gw
                    v_w = beta2 * v_w + (1 - beta2) * gw * gw
                    m_b = beta1 * m_b + (1 - beta1) * gb
                    v_b = beta2 * v_b + (1 - beta2) * gb * gb
                    m_w_hat = m_w / (1 - beta1**step_idx)
                    v_w_hat = v_w / (1 - beta2**step_idx)
                    m_b_hat = m_b / (1 - beta1**step_idx)
                    v_b_hat = v_b / (1 - beta2**step_idx)
                    w = w - self.lr * m_w_hat / (torch.sqrt(v_w_hat) + eps)
                    b = b - self.lr * m_b_hat / (torch.sqrt(v_b_hat) + eps)
                else:
                    w = w - self.lr * gw
                    b = b - self.lr * gb
                if self.decay:
                    w = w * (1.0 - self.decay)
                w = torch.nan_to_num(w, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
                b = torch.nan_to_num(b, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
            w = w.detach().requires_grad_(True)
            b = b.detach().requires_grad_(True)
            last_loss = float(loss.detach().item())
        state.params = {"weight": w.detach(), "bias": b.detach()}
        state.updates += 1
        state.last_loss = last_loss
        return state

    def state_bytes(self, state: FastMemoryState) -> int:
        return state.state_bytes()


class TTTMLP(nn.Module):
    """Optional two-layer session-local TTT-MLP.

    More expressive than TTT-Linear but substantially more expensive to adapt.
    This is an experimental implementation, not a faithful reproduction of a
    published architecture.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        lr: float = 0.02,
        steps: int = 1,
        clip: float = 1.0,
        weight_decay: float = 0.0,
        decay: float = 0.0,
        init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.steps = steps
        self.clip = clip
        self.weight_decay = weight_decay
        self.decay = decay
        self.register_buffer("init_w1", torch.randn(hidden_dim, in_dim) * init_scale)
        self.register_buffer("init_b1", torch.zeros(hidden_dim))
        self.register_buffer("init_w2", torch.randn(out_dim, hidden_dim) * init_scale)
        self.register_buffer("init_b2", torch.zeros(out_dim))

    def init_state(self, device: torch.device | str = "cpu") -> FastMemoryState:
        return FastMemoryState(
            kind="ttt_mlp",
            params={
                "w1": self.init_w1.detach().clone().to(device),
                "b1": self.init_b1.detach().clone().to(device),
                "w2": self.init_w2.detach().clone().to(device),
                "b2": self.init_b2.detach().clone().to(device),
            },
        )

    @staticmethod
    def functional_forward(
        params: dict[str, torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        h = F.gelu(F.linear(x, params["w1"], params["b1"]))
        return F.linear(h, params["w2"], params["b2"])

    def read(self, state: FastMemoryState, queries: torch.Tensor) -> torch.Tensor:
        if state.kind != "ttt_mlp":
            raise ValueError(f"state kind {state.kind!r} cannot be read by TTTMLP")
        return self.functional_forward(state.params, queries)

    def adapt(
        self,
        state: FastMemoryState,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> FastMemoryState:
        if state.kind != "ttt_mlp":
            raise ValueError(f"state kind {state.kind!r} cannot be adapted by TTTMLP")
        if keys.ndim != 2 or values.ndim != 2:
            raise ValueError("keys and values must be [N, dim]")
        keys = torch.nan_to_num(keys.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        values = torch.nan_to_num(values.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        p = {k: v.detach().clone().requires_grad_(True) for k, v in state.params.items()}
        last_loss = 0.0
        for _ in range(max(1, self.steps)):
            h = F.gelu(F.linear(keys, p["w1"], p["b1"]))
            pred = F.linear(h, p["w2"], p["b2"])
            loss = F.mse_loss(pred, values)
            grads = torch.autograd.grad(
                loss, (p["w1"], p["b1"], p["w2"], p["b2"]), allow_unused=False
            )
            with torch.no_grad():
                for name, g in zip(("w1", "b1", "w2", "b2"), grads):
                    g = _clip_tensor(g, self.clip)
                    if self.weight_decay and name.startswith("w"):
                        g = g + self.weight_decay * p[name]
                    p[name] = p[name] - self.lr * g
                    if self.decay:
                        p[name] = p[name] * (1.0 - self.decay)
                    p[name] = torch.nan_to_num(
                        p[name], nan=0.0, posinf=10.0, neginf=-10.0
                    ).clamp(-10, 10)
            p = {k: v.detach().requires_grad_(True) for k, v in p.items()}
            last_loss = float(loss.detach().item())
        state.params = {k: v.detach() for k, v in p.items()}
        state.updates += 1
        state.last_loss = last_loss
        return state


class AssociativeMemory(nn.Module):
    """Differentiable key-value fast-weight matrix with a delta rule.

    This is intentionally a separate baseline from TTT-Linear.  It stores
    associations by an error-correcting outer-product update.
    """

    def __init__(
        self,
        dim: int,
        lr: float = 0.5,
        decay: float = 0.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.lr = lr
        self.decay = decay
        self.normalize = normalize
        self.register_buffer("init_weight", torch.zeros(dim, dim))

    def init_state(self, device: torch.device | str = "cpu") -> FastMemoryState:
        return FastMemoryState(
            kind="associative",
            params={"weight": self.init_weight.detach().clone().to(device)},
        )

    def read(self, state: FastMemoryState, queries: torch.Tensor) -> torch.Tensor:
        if state.kind != "associative":
            raise ValueError("state is not an associative memory state")
        return F.linear(queries, state.params["weight"])

    def adapt(
        self,
        state: FastMemoryState,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> FastMemoryState:
        if state.kind != "associative":
            raise ValueError("state is not an associative memory state")
        if keys.ndim != 2 or values.ndim != 2:
            raise ValueError("keys and values must be [N, dim]")
        k = keys.detach()
        v = values.detach()
        if self.normalize:
            k = F.normalize(k, dim=-1)
            v = F.normalize(v, dim=-1)
        w = state.params["weight"].detach().clone()
        pred = F.linear(k, w)
        error = v - pred
        update = error.t() @ k / max(1, k.shape[0])
        w = w + self.lr * update
        if self.decay:
            w = w * (1.0 - self.decay)
        w = torch.nan_to_num(w, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
        state.params = {"weight": w}
        state.updates += 1
        state.last_loss = float(error.pow(2).mean().item())
        return state


class FastWeightMemory(nn.Module):
    """Stateless dispatcher for a session-local fast-weight state."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.kind = cfg.memory_kind
        self.cfg = cfg
        if self.kind == "ttt_linear":
            self.backend = TTTLinear(
                cfg.memory_dim,
                cfg.memory_dim,
                lr=cfg.memory_inner_lr,
                steps=cfg.memory_inner_steps,
                clip=cfg.memory_clip,
                weight_decay=cfg.memory_weight_decay,
                decay=cfg.memory_decay,
                inner_optimizer=cfg.memory_inner_optimizer,
            )
        elif self.kind == "ttt_mlp":
            self.backend = TTTMLP(
                cfg.memory_dim,
                cfg.memory_dim,
                hidden_dim=cfg.memory_dim * cfg.memory_hidden_mult,
                lr=cfg.memory_inner_lr,
                steps=cfg.memory_inner_steps,
                clip=cfg.memory_clip,
                weight_decay=cfg.memory_weight_decay,
                decay=cfg.memory_decay,
            )
        elif self.kind == "associative":
            self.backend = AssociativeMemory(
                cfg.memory_dim,
                lr=max(cfg.memory_inner_lr, 0.1),
                decay=cfg.memory_decay,
            )
        else:  # pragma: no cover - config validates this
            raise ValueError(f"unknown memory kind: {self.kind}")

    def init_state(self, device: torch.device | str = "cpu") -> FastMemoryState:
        return self.backend.init_state(device)

    def read(self, state: FastMemoryState, queries: torch.Tensor) -> torch.Tensor:
        return self.backend.read(state, queries)

    def adapt(
        self,
        state: FastMemoryState,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> FastMemoryState:
        start = time.perf_counter()
        state = self.backend.adapt(state, keys, values)
        state.adapt_seconds += time.perf_counter() - start
        return state

    def state_bytes(self, state: FastMemoryState) -> int:
        return state.state_bytes()
