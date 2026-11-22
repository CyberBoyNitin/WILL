"""Request schemas for the WILL API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ExecutionMode = Literal["fixed_depth", "adaptive_soft", "adaptive_hard"]
TaskName = Literal[
    "lm", "arithmetic", "arith_add", "multihop", "sequence", "longctx", "mixture"
]


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    checkpoint_id: str | None = None
    mode: ExecutionMode = "adaptive_hard"
    max_steps: int | None = Field(default=None, ge=1, le=64)
    halt_epsilon: float | None = Field(default=None, gt=0.0, lt=1.0)
    memory_session_id: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=4.0)
    top_k: int = Field(default=0, ge=0, le=260)
    max_new_tokens: int = Field(default=32, ge=1, le=256)
    append_sep: bool = True


class ForwardRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    checkpoint_id: str | None = None
    mode: ExecutionMode = "adaptive_soft"
    max_steps: int | None = Field(default=None, ge=1, le=64)
    halt_epsilon: float | None = Field(default=None, gt=0.0, lt=1.0)
    top_k: int = Field(default=5, ge=1, le=20)


class TrainRequest(BaseModel):
    task: TaskName = "arith_add"
    profile: Literal["tiny", "standard", "experimental", "custom"] = "tiny"
    epochs: int = Field(default=3, ge=1, le=500)
    train_size: int = Field(default=1000, ge=1, le=200000)
    val_size: int = Field(default=150, ge=1, le=20000)
    test_size: int = Field(default=150, ge=1, le=20000)
    batch_size: int = Field(default=16, ge=1, le=512)
    lr: float = Field(default=5e-4, gt=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    grad_clip: float = Field(default=1.0, ge=0.0, le=100.0)
    grad_accum: int = Field(default=1, ge=1, le=64)
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    scheduler: Literal["cosine", "linear", "none"] = "cosine"
    warmup_steps: int = Field(default=20, ge=0, le=100000)
    ponder_lambda: float = Field(default=0.0, ge=0.0, le=10.0)
    ponder_warmup_epochs: int = Field(default=0, ge=0, le=500)
    memory_lambda: float = Field(default=0.0, ge=0.0, le=10.0)
    mode: Literal["fixed_depth", "adaptive_soft"] = "fixed_depth"
    precision: Literal["fp32", "bf16", "fp16", "auto"] = "fp32"
    torch_compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = (
        "reduce-overhead"
    )
    seed: int = Field(default=1337, ge=0, le=2**31 - 1)
    eval_every: int = Field(default=1, ge=1, le=100)
    early_stop_patience: int = Field(default=0, ge=0, le=1000)
    max_steps: int | None = Field(default=None, ge=1, le=64)
    min_steps: int = Field(default=1, ge=1, le=64)
    halt_epsilon: float = Field(default=0.05, gt=0.0, lt=1.0)
    min_complexity: int = Field(default=1, ge=1, le=4)
    max_complexity: int = Field(default=2, ge=1, le=4)
    mixed_easy_ratio: float | None = Field(default=None, gt=0.0, lt=1.0)
    checkpoint_name: str | None = Field(default=None, max_length=96)
    device: str = "auto"

    @model_validator(mode="after")
    def _validate(self) -> "TrainRequest":
        if self.max_complexity < self.min_complexity:
            raise ValueError("max_complexity must be >= min_complexity")
        if self.max_steps is not None and self.min_steps > self.max_steps:
            raise ValueError("min_steps cannot exceed max_steps")
        return self


class BenchmarkRequest(BaseModel):
    suite: Literal[
        "reasoning",
        "adaptive_compute",
        "mixed_difficulty",
        "memory_recall",
        "continual",
        "inference",
        "scaling",
        "ablation",
    ]
    checkpoint_id: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class MemoryCreateRequest(BaseModel):
    name: str = Field(default="session", min_length=1, max_length=120)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryTextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000_000)
    filename: str = Field(default="inline.txt", max_length=200)


class MemoryQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: Literal["lexical", "neural", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=50)
    use_memory_in_generation: bool = True


class MemoryAdaptRequest(BaseModel):
    rounds: int = Field(default=1, ge=1, le=200)


class MemorySearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: Literal["lexical", "neural", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=50)


class ChatMessage(BaseModel):
    role: Literal[
        "system", "developer", "user", "assistant", "tool", "function"
    ] = "user"
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request.

    ``model`` accepts a WILL checkpoint id as well as the aliases ``will`` and
    ``will-engine``.  The extra ``mode``, ``max_steps``, ``memory_session_id``
    and ``chat_template`` fields are WILL-specific additions; standard OpenAI
    clients can ignore them.
    """

    model: str = Field(default="will", max_length=200)
    messages: list[ChatMessage] = Field(min_length=1, max_length=200)
    stream: bool = False
    temperature: float = Field(default=0.0, ge=0.0, le=4.0)
    top_k: int = Field(default=0, ge=0, le=260)
    max_tokens: int | None = Field(default=None, ge=1, le=512)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=512)
    stop: str | list[str] | None = None
    mode: ExecutionMode = "adaptive_hard"
    max_steps: int | None = Field(default=None, ge=1, le=64)
    halt_epsilon: float | None = Field(default=None, gt=0.0, lt=1.0)
    memory_session_id: str | None = None
    checkpoint_id: str | None = None
    chat_template: Literal["plain", "role"] = "plain"
    stream_options: dict[str, Any] | None = None

    @property
    def max_new_tokens(self) -> int:
        return int(self.max_completion_tokens or self.max_tokens or 32)


class MemoryAbsorbRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000_000)
    session_id: str | None = None
    name: str = Field(default="api-memory", min_length=1, max_length=120)
    filename: str = Field(default="inline.txt", max_length=200)
    consolidate: bool = False
    checkpoint_id: str | None = None


class OpenAIMemoryQueryRequest(BaseModel):
    session_id: str = Field(min_length=32, max_length=32)
    question: str = Field(min_length=1, max_length=4000)
    mode: Literal["lexical", "neural", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=50)
    checkpoint_id: str | None = None
