"""WILL engine SDK.

``WILLForCausalLM`` wraps a checkpoint, byte tokenizer, session-local fast
memory and process telemetry behind a small API.  The FastAPI server uses the
same class for its OpenAI-compatible routes.
"""
from __future__ import annotations

from will_engine.chat import CHAT_TEMPLATES, format_chat_prompt
from will_engine.model import WILLForCausalLM, estimate_flops
from will_engine.telemetry import GLOBAL_TELEMETRY, Telemetry
from will_engine.types import GenerationResult

__version__ = "0.2.0"

__all__ = [
    "CHAT_TEMPLATES",
    "GLOBAL_TELEMETRY",
    "GenerationResult",
    "Telemetry",
    "WILLForCausalLM",
    "estimate_flops",
    "format_chat_prompt",
]
