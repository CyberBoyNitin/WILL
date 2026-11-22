"""Chat prompt formatting for the OpenAI-compatible transport.

WILL checkpoints are task models, not instruction-tuned chat models.  The
default ``plain`` template therefore passes message text through unchanged
(joined by newlines).  ``role`` adds role prefixes for experiments that were
trained with them; it is opt-in and never claimed to be a universal chat
template.
"""
from __future__ import annotations

from typing import Any, Iterable

CHAT_TEMPLATES = ("plain", "role")


def message_text(content: Any) -> str:
    """Flatten OpenAI-style string or content-part arrays to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") in (None, "text") and "text" in part:
                    parts.append(str(part["text"]))
        return "".join(parts)
    return str(content)


def format_chat_prompt(
    messages: Iterable[Any], template: str = "plain"
) -> str:
    """Turn messages into a single causal-LM prompt.

    ``messages`` may contain objects with ``role``/``content`` attributes or
    plain dicts, so the same helper serves the SDK and the HTTP route.
    """
    if template not in CHAT_TEMPLATES:
        raise ValueError(
            f"unknown chat template {template!r}; expected one of {CHAT_TEMPLATES}"
        )
    rows: list[tuple[str, str]] = []
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role") or "user")
            content = message.get("content")
        else:
            role = str(getattr(message, "role", "user"))
            content = getattr(message, "content", "")
        text = message_text(content)
        if text:
            rows.append((role, text))
    if not rows:
        raise ValueError("messages must contain at least one non-empty content")
    if template == "role":
        body = "\n".join(f"{role}: {text}" for role, text in rows)
        return body + "\nassistant:"
    return "\n".join(text for _, text in rows)
