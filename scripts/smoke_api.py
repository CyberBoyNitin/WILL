"""Smoke-test the public OpenAI-compatible WILL API on 127.0.0.1:8000.

Start the server first::

    will-serve --host 127.0.0.1 --port 8000

Then run::

    python scripts/smoke_api.py
"""
from __future__ import annotations

import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=60) as response:
        return json.load(response)


def post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> int:
    checks: dict[str, object] = {}
    try:
        checks["health"] = get("/api/health")
        models = get("/v1/models")
        checks["models"] = {
            "object": models.get("object"),
            "count": len(models.get("data", [])),
            "aliases": [item["id"] for item in models.get("data", [])[:1]],
        }
        checks["completion"] = post(
            "/v1/chat/completions",
            {
                "model": "will",
                "messages": [{"role": "user", "content": "calc 2 + 3 ="}],
                "mode": "fixed_depth",
                "max_steps": 1,
                "max_new_tokens": 6,
            },
        )["choices"][0]["message"]
        absorbed = post(
            "/v1/memory/absorb",
            {"text": "WILL API smoke test: the capital of France is Paris."},
        )
        checks["absorb"] = {
            "session_id": absorbed.get("session_id"),
            "chunks_added": absorbed.get("chunks_added"),
            "checkpoint_id": absorbed.get("checkpoint_id"),
        }
        telemetry = get("/v1/telemetry")
        checks["telemetry"] = {
            "format": telemetry.get("format"),
            "generation": telemetry.get("generation", {}).get("requests"),
            "default_checkpoint_id": telemetry.get("default_checkpoint_id"),
        }
    except Exception as exc:  # noqa: BLE001 - smoke test reports any failure
        print(json.dumps({"ok": False, "error": repr(exc), "checks": checks}, indent=2))
        return 1
    print(json.dumps({"ok": True, "checks": checks}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
