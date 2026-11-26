"""WILL engine SDK.

``WILLForCausalLM`` wraps a checkpoint, byte tokenizer, session-local fast
memory and process telemetry behind a small API.  The FastAPI server uses the
same class for its OpenAI-compatible routes.
"""
from __future__ import annotations

from backend.config import AppConfig as WILLServerConfig
from backend.config import ModelConfig as WILLConfig
from backend.engine.model import WILLModel
from will_engine.chat import CHAT_TEMPLATES, format_chat_prompt
from will_engine.model import WILLForCausalLM, estimate_flops
from will_engine.telemetry import GLOBAL_TELEMETRY, Telemetry
from will_engine.types import GenerationResult

__version__ = "0.3.0"

__all__ = [
    "CHAT_TEMPLATES",
    "GLOBAL_TELEMETRY",
    "GenerationResult",
    "Telemetry",
    "WILLConfig",
    "WILLForCausalLM",
    "WILLModel",
    "WILLServerConfig",
    "estimate_flops",
    "format_chat_prompt",
]
