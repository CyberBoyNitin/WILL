"""Central configuration for WILL.

The config is deliberately small, serialisable and reproducible.  Every
training run and benchmark stores the resolved configuration alongside its
results so that numbers can be traced back to an exact model definition.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[1]


def _is_source_checkout() -> bool:
    """True when ``backend`` sits next to this repository's pyproject.toml."""
    return (ROOT / "pyproject.toml").is_file() and (ROOT / "backend").is_dir()


class ModelConfig(BaseModel):
    """Neural architecture configuration."""

    vocab_size: int = 260
    hidden_dim: int = 128
    n_heads: int = 4
    context_layers: int = 1
    context_ff_mult: int = 2
    max_seq_len: int = 256
    dropout: float = 0.1

    # Adaptive computation
    min_steps: int = 1
    max_steps: int = 4
    # Deployment cap for adaptive inference.  The model can still be trained
    # with ``max_steps``; on held-out data the reference checkpoint reaches the
    # same accuracy with a three-step cap, so the fourth step is optional.
    adaptive_inference_steps: int | None = 3
    halt_epsilon: float = 0.05
    ff_mult: int = 2

    # Fast-weight memory
    memory_dim: int = 64
    memory_kind: Literal["ttt_linear", "ttt_mlp", "associative"] = "ttt_linear"
    memory_inner_lr: float = 0.05
    memory_inner_steps: int = 2
    memory_inner_optimizer: Literal["sgd", "adam"] = "adam"
    memory_update_every: int = 8
    memory_decay: float = 0.0
    memory_clip: float = 1.0
    memory_weight_decay: float = 0.0
    memory_norm_max: float = 0.0
    memory_ridge: float = 1e-3
    memory_hidden_mult: int = 2
    # Sequential answer-slot memory.  Each record can store one value per
    # answer token position, which avoids decoding a whole answer from one
    # pooled vector.
    memory_slots: int = 4
    # Add retrieved token-embedding similarities directly to the output logits.
    memory_logit_skip: bool = False
    memory_logit_scale: float = 20.0
    # Mix a content-derived bag-of-tokens key into the pooled record key.  The
    # contextual key can collapse to nearly the same vector for short prompts;
    # token identity keeps distinct questions addressable before training.
    memory_lexical_scale: float = 0.0

    tie_embeddings: bool = False

    @field_validator("hidden_dim")
    @classmethod
    def _hidden_positive(cls, v: int) -> int:
        if v < 16:
            raise ValueError("hidden_dim must be >= 16")
        return v

    @field_validator("n_heads")
    @classmethod
    def _heads_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_heads must be >= 1")
        return v

    @model_validator(mode="after")
    def _validate(self) -> "ModelConfig":
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if self.min_steps < 1:
            raise ValueError("min_steps must be >= 1")
        if self.max_steps < self.min_steps:
            raise ValueError("max_steps must be >= min_steps")
        if self.max_steps > 64:
            raise ValueError("max_steps > 64 is not supported")
        if self.adaptive_inference_steps is not None:
            if self.adaptive_inference_steps < 1:
                raise ValueError("adaptive_inference_steps must be >= 1")
            if self.adaptive_inference_steps > self.max_steps:
                self.adaptive_inference_steps = self.max_steps
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.halt_epsilon < 1.0:
            raise ValueError("halt_epsilon must be in (0, 1)")
        if self.context_layers < 0:
            raise ValueError("context_layers must be >= 0")
        if self.memory_dim < 8:
            raise ValueError("memory_dim must be >= 8")
        if self.memory_norm_max < 0:
            raise ValueError("memory_norm_max must be >= 0")
        if self.memory_ridge < 0:
            raise ValueError("memory_ridge must be >= 0")
        if self.max_seq_len < 8:
            raise ValueError("max_seq_len must be >= 8")
        return self

    def estimate_parameter_count(self) -> int:
        """Cheap analytic estimate used for pre-flight memory warnings."""
        d = self.hidden_dim
        v = self.vocab_size
        emb = v * d + self.max_seq_len * d
        ctx = self.context_layers * (
            4 * d * d + 2 * d * (self.context_ff_mult * d)
        )
        block = (
            4 * d * d
            + 2 * d * (self.ff_mult * d)
            + self.max_steps * d
        )
        halt = 3 * d * (d // 2) + (d // 2) + 2 * d * d
        mem = 2 * d * self.memory_dim + self.memory_dim * d
        head = d * v
        return int(emb + ctx + block + halt + mem + head)

    def resolved_adaptive_steps(self) -> int:
        """Inference depth cap for adaptive modes (training may use more)."""
        if self.adaptive_inference_steps is None:
            return self.max_steps
        return max(1, min(self.max_steps, self.adaptive_inference_steps))


class TrainingConfig(BaseModel):
    task: Literal[
        "lm", "arithmetic", "arith_add", "multihop", "sequence", "longctx", "mixture"
    ] = "arithmetic"
    train_size: int = 1500
    val_size: int = 200
    test_size: int = 200
    epochs: int = 3
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    grad_accum: int = 1
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    scheduler: Literal["cosine", "linear", "none"] = "cosine"
    warmup_steps: int = 20
    ponder_lambda: float = 0.02
    ponder_warmup_epochs: int = 0
    accuracy_gated_ponder: bool = False
    memory_lambda: float = 0.0
    mode: Literal["fixed_depth", "adaptive_soft"] = "adaptive_soft"
    seed: int = 1337
    eval_every: int = 1
    early_stop_patience: int = 0
    max_steps: int | None = None
    min_complexity: int = 1
    max_complexity: int = 2
    # When set, train/val/test use an exact mixed-difficulty composition:
    # ``mixed_easy_ratio`` of examples at min_complexity, the rest at
    # max_complexity.  ``None`` keeps the existing uniform sampler.
    mixed_easy_ratio: float | None = None
    precision: Literal["fp32", "bf16", "fp16", "auto"] = "fp32"
    torch_compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
    profile: Literal["tiny", "standard", "experimental", "custom"] = "tiny"
    checkpoint_name: str | None = None

    @field_validator("batch_size", "train_size", "val_size", "test_size")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @field_validator("min_complexity", "max_complexity")
    @classmethod
    def _complexity(cls, v: int) -> int:
        if v < 1:
            raise ValueError("complexity must be >= 1")
        return v

    @field_validator("ponder_warmup_epochs")
    @classmethod
    def _ponder_warmup(cls, v: int) -> int:
        if v < 0:
            raise ValueError("ponder_warmup_epochs must be >= 0")
        return v

    @field_validator("mixed_easy_ratio")
    @classmethod
    def _mixed_ratio(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 < v < 1.0:
            raise ValueError("mixed_easy_ratio must be in (0, 1)")
        return v

    @model_validator(mode="after")
    def _validate_complexity(self) -> "TrainingConfig":
        if self.max_complexity < self.min_complexity:
            raise ValueError("max_complexity must be >= min_complexity")
        return self

    @field_validator("lr")
    @classmethod
    def _lr(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("lr must be in (0, 1]")
        return v


class MemoryConfig(BaseModel):
    chunk_size: int = 192
    chunk_overlap: int = 32
    max_upload_bytes: int = 12 * 1024 * 1024
    top_k: int = 5
    window: int = 24
    stride: int = 12
    lexical_weight: float = 0.6
    neural_weight: float = 0.4
    min_retrieval_score: float = 0.08
    max_answer_tokens: int = 24
    ttt_lr: float = 0.05
    ttt_steps: int = 2
    max_chunks: int = 400

    @model_validator(mode="after")
    def _validate(self) -> "MemoryConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        total = self.lexical_weight + self.neural_weight
        if total <= 0:
            raise ValueError("retrieval weights must be positive")
        self.lexical_weight /= total
        self.neural_weight /= total
        return self


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    api_key: str | None = None
    device: str = "auto"
    data_dir: str = str(ROOT / "data")
    checkpoint_dir: str = str(ROOT / "checkpoints")
    experiment_dir: str = str(ROOT / "experiments")
    upload_dir: str = str(ROOT / "data" / "uploads")
    memory_dir: str = str(ROOT / "data" / "memory_sessions")
    db_path: str = str(ROOT / "data" / "will.db")
    log_level: str = "INFO"


class AppConfig(BaseModel):
    profile: str = "tiny"
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    def ensure_dirs(self) -> None:
        for p in (
            self.server.data_dir,
            self.server.checkpoint_dir,
            self.server.experiment_dir,
            self.server.upload_dir,
            self.server.memory_dir,
        ):
            Path(p).mkdir(parents=True, exist_ok=True)


PROFILE_OVERRIDES: dict[str, dict] = {
    "tiny": {
        "model": {
            "hidden_dim": 128,
            "n_heads": 4,
            "context_layers": 1,
            "context_ff_mult": 2,
            "max_steps": 4,
            "ff_mult": 2,
            "memory_dim": 64,
        },
        "training": {"batch_size": 16},
    },
    "standard": {
        "model": {
            "hidden_dim": 256,
            "n_heads": 4,
            "context_layers": 2,
            "context_ff_mult": 4,
            "max_steps": 8,
            "ff_mult": 4,
            "memory_dim": 128,
        },
        "training": {"batch_size": 32},
    },
}


# Parameter-matched scaling presets.  The exact counts are verified by
# instantiating the model in the scaling benchmark; ``target_parameters`` is
# only the design target, never a reported measurement.
SCALING_PRESETS: dict[str, dict] = {
    "15m": {
        "target_parameters": 15_000_000,
        "model": {
            "vocab_size": 260,
            "hidden_dim": 624,
            "n_heads": 8,
            "context_layers": 2,
            "context_ff_mult": 2,
            "max_seq_len": 256,
            "max_steps": 4,
            "min_steps": 1,
            "ff_mult": 4,
            "memory_dim": 128,
            "dropout": 0.0,
        },
    },
    "50m": {
        "target_parameters": 50_000_000,
        "model": {
            "vocab_size": 260,
            "hidden_dim": 1152,
            "n_heads": 8,
            "context_layers": 2,
            "context_ff_mult": 2,
            "max_seq_len": 256,
            "max_steps": 4,
            "min_steps": 1,
            "ff_mult": 4,
            "memory_dim": 128,
            "dropout": 0.0,
        },
    },
    "125m": {
        "target_parameters": 125_000_000,
        "model": {
            "vocab_size": 260,
            "hidden_dim": 1824,
            "n_heads": 8,
            "context_layers": 2,
            "context_ff_mult": 2,
            "max_seq_len": 256,
            "max_steps": 4,
            "min_steps": 1,
            "ff_mult": 4,
            "memory_dim": 128,
            "dropout": 0.0,
        },
    },
}


def scaling_model_config(name: str) -> ModelConfig:
    """Return a fresh :class:`ModelConfig` for a named scaling preset."""
    if name not in SCALING_PRESETS:
        raise ValueError(
            f"unknown scaling preset {name!r}; expected one of {tuple(SCALING_PRESETS)}"
        )
    return ModelConfig(**SCALING_PRESETS[name]["model"])


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_profile(cfg: AppConfig, profile: str) -> AppConfig:
    """Return a copy of ``cfg`` with a named profile's defaults applied."""
    if profile == cfg.profile:
        return cfg
    if profile not in PROFILE_OVERRIDES and profile not in ("experimental", "custom"):
        raise ValueError(f"unknown profile: {profile}")
    merged = _deep_merge(cfg.model_dump(), PROFILE_OVERRIDES.get(profile, {}))
    merged["profile"] = profile
    return AppConfig(**merged)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML config, applying profile defaults and env overrides."""
    raw: dict = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    profile = raw.get("profile", "tiny")
    merged = _deep_merge({"profile": profile}, PROFILE_OVERRIDES.get(profile, {}))
    merged = _deep_merge(merged, raw)
    cfg = AppConfig(**merged)

    if os.environ.get("WILL_API_KEY"):
        cfg.server.api_key = os.environ["WILL_API_KEY"]
    if os.environ.get("WILL_DATA_DIR"):
        d = os.environ["WILL_DATA_DIR"]
        cfg.server.data_dir = d
        cfg.server.upload_dir = str(Path(d) / "uploads")
        cfg.server.memory_dir = str(Path(d) / "memory_sessions")
        cfg.server.db_path = str(Path(d) / "will.db")
    if os.environ.get("WILL_CHECKPOINT_DIR"):
        cfg.server.checkpoint_dir = os.environ["WILL_CHECKPOINT_DIR"]
    if os.environ.get("WILL_EXPERIMENT_DIR"):
        cfg.server.experiment_dir = os.environ["WILL_EXPERIMENT_DIR"]
    if os.environ.get("WILL_DEVICE"):
        cfg.server.device = os.environ["WILL_DEVICE"]
    if not _is_source_checkout():
        # An installed wheel lives under site-packages, which may be read-only.
        # Default writable state to ``WILL_HOME`` or the current directory
        # instead of trying to create data/ inside the package directory.
        base = Path(os.environ.get("WILL_HOME", Path.cwd()))
        server_raw = raw.get("server") or {}
        data_explicit = any(
            key in server_raw
            for key in ("data_dir", "upload_dir", "memory_dir", "db_path")
        )
        if not os.environ.get("WILL_DATA_DIR") and not data_explicit:
            cfg.server.data_dir = str(base / "data")
            cfg.server.upload_dir = str(base / "data" / "uploads")
            cfg.server.memory_dir = str(base / "data" / "memory_sessions")
            cfg.server.db_path = str(base / "data" / "will.db")
        if (
            not os.environ.get("WILL_CHECKPOINT_DIR")
            and "checkpoint_dir" not in server_raw
        ):
            cfg.server.checkpoint_dir = str(base / "checkpoints")
        if (
            not os.environ.get("WILL_EXPERIMENT_DIR")
            and "experiment_dir" not in server_raw
        ):
            cfg.server.experiment_dir = str(base / "experiments")
    cfg.ensure_dirs()
    return cfg


def save_config(cfg: AppConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(), f, sort_keys=False)
