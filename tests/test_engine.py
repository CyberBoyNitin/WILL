"""SDK and OpenAI-compatible endpoint tests."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.engine.model import WILLModel
from backend.engine.tokenizer import ByteTokenizer
from backend.server.app import create_app
from backend.training.trainer import Trainer
from will_engine import WILLForCausalLM


@pytest.fixture()
def trained_app(temp_app_cfg):
    tokenizer = ByteTokenizer()
    model = WILLModel(temp_app_cfg.model)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        model_cfg=temp_app_cfg.model,
        training_cfg=temp_app_cfg.training,
        checkpoint_dir=temp_app_cfg.server.checkpoint_dir,
        device="cpu",
        run_name="api-test",
    )
    trainer.train()
    app = create_app(temp_app_cfg)
    with TestClient(app) as client:
        yield client, "api-test-final"


def test_sdk_generate_stream_and_telemetry(temp_app_cfg):
    # Build the checkpoint once without the HTTP layer.
    model = WILLModel(temp_app_cfg.model)
    Trainer(
        model=model,
        tokenizer=ByteTokenizer(),
        model_cfg=temp_app_cfg.model,
        training_cfg=temp_app_cfg.training,
        checkpoint_dir=temp_app_cfg.server.checkpoint_dir,
        device="cpu",
        run_name="sdk-test",
    ).train()
    engine = WILLForCausalLM(
        "sdk-test-final",
        checkpoint_dir=temp_app_cfg.server.checkpoint_dir,
        config=temp_app_cfg,
        memory_dir=temp_app_cfg.server.memory_dir,
    )
    result = engine.generate("calc 2 + 3 =", max_new_tokens=4)
    assert isinstance(result.text, str)
    assert result.usage["total_tokens"] == (
        result.usage["prompt_tokens"] + result.usage["completion_tokens"]
    )
    assert result.finish_reason in {"stop", "length"}
    events = list(engine.stream("calc 7 + 1 =", max_new_tokens=4))
    assert events[0]["type"] == "start"
    assert any(event["type"] == "token" for event in events)
    assert events[-1]["type"] == "done"
    telemetry = engine.telemetry(include_hardware=False)
    assert telemetry["generation"]["requests"] >= 2
    assert telemetry["generation"]["streaming_requests"] >= 1
    assert telemetry["energy_measured"] is False


def test_openai_chat_completion_and_stream(trained_app):
    client, checkpoint_id = trained_app
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": checkpoint_id,
            "messages": [{"role": "user", "content": "calc 2 + 3 ="}],
            "max_tokens": 4,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["total_tokens"] > 0
    assert body["will"]["checkpoint_id"] == checkpoint_id

    stream = client.post(
        "/v1/chat/completions",
        json={
            "model": "will",
            "checkpoint_id": checkpoint_id,
            "messages": [{"role": "user", "content": "calc 7 + 1 ="}],
            "max_tokens": 4,
            "stream": True,
        },
    )
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")
    lines = [line for line in stream.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert any(chunk["choices"][0]["delta"].get("content") for chunk in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] in {"stop", "length"}


def test_openai_memory_absorb_query_and_telemetry(trained_app):
    client, checkpoint_id = trained_app
    absorb = client.post(
        "/v1/memory/absorb",
        json={
            "text": "The east vault access code is 4815. The backup code is 9021.",
            "name": "openai-test",
        },
    )
    assert absorb.status_code == 200
    body = absorb.json()
    assert body["chunks_added"] >= 1
    session_id = body["session_id"]
    assert len(session_id) == 32

    chat = client.post(
        "/v1/chat/completions",
        json={
            "model": checkpoint_id,
            "messages": [{"role": "user", "content": "calc 2 + 3 ="}],
            "max_tokens": 2,
            "memory_session_id": session_id,
        },
    )
    assert chat.status_code == 200

    query = client.post(
        "/v1/memory/query",
        json={
            "session_id": session_id,
            "question": "what is the east vault access code?",
        },
    )
    assert query.status_code == 200
    assert "sources" in query.json()

    telemetry = client.get("/v1/telemetry")
    assert telemetry.status_code == 200
    payload = telemetry.json()
    assert payload["object"] == "will.telemetry"
    assert payload["generation"]["requests"] >= 1
    assert payload["memory"]["absorb_calls"] >= 1
    assert payload["energy_measured"] is False


def test_memory_session_checkpoint_mismatch_is_rejected(temp_app_cfg):
    model = WILLModel(temp_app_cfg.model)
    engine = WILLForCausalLM(
        "checkpoint-a",
        config=temp_app_cfg,
        model=model,
        memory_dir=temp_app_cfg.server.memory_dir,
    )
    session = engine.memory_manager.create_session(
        "mismatch", metadata={"checkpoint_id": "checkpoint-b"}
    )
    with pytest.raises(ValueError, match="belongs to checkpoint"):
        engine.generate(
            "calc 2 + 3 =",
            max_new_tokens=1,
            memory_session_id=session.session_id,
        )


def test_api_key_guard_covers_v1(temp_app_cfg):
    temp_app_cfg.server.api_key = "secret"
    app = create_app(temp_app_cfg)
    with TestClient(app) as client:
        assert client.get("/v1/models").status_code == 401
        assert (
            client.get("/v1/models", headers={"x-api-key": "secret"}).status_code
            == 200
        )
