"""API integration tests against real services and real jobs."""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from backend.server.app import create_app


def _wait_job(client, path: str, job_id: str, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"{path}/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()["job"]
        if job["status"] in {"completed", "failed", "cancelled", "interrupted"}:
            return job
        time.sleep(0.25)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_system_and_dataset_endpoints(client):
    for path in (
        "/api/health",
        "/api/system",
        "/api/config",
        "/api/overview",
        "/api/checkpoints",
        "/api/datasets",
        "/api/benchmarks/suites",
        "/api/experiments",
        "/api/training/defaults",
    ):
        response = client.get(path)
        assert response.status_code == 200, (path, response.text)
    sample = client.get("/api/datasets/arith_add/sample?n=3")
    assert sample.status_code == 200
    body = sample.json()
    assert len(body["examples"]) == 3
    assert all("prompt" in ex and "answer" in ex for ex in body["examples"])


def test_playground_requires_a_checkpoint(client):
    response = client.post(
        "/api/playground/generate",
        json={"prompt": "calc 1 + 2 =", "max_new_tokens": 4},
    )
    assert response.status_code == 400
    tokenized = client.post("/api/playground/tokenize", json={"text": "hello"})
    assert tokenized.status_code == 200
    assert tokenized.json()["token_count"] == 5


def test_memory_session_lifecycle(client):
    created = client.post("/api/memory/sessions", json={"name": "api-memory"})
    assert created.status_code == 200, created.text
    session_id = created.json()["session_id"]

    ingest = client.post(
        f"/api/memory/sessions/{session_id}/ingest-text",
        json={
            "text": "K00=V100. K01=V101. access code alpha_1234.",
            "filename": "facts.txt",
        },
    )
    assert ingest.status_code == 200, ingest.text
    assert ingest.json()["ingest"]["chunks_added"] >= 1

    query = client.post(
        f"/api/memory/sessions/{session_id}/query",
        json={"question": "what is the value of K00?", "mode": "hybrid"},
    )
    assert query.status_code == 200, query.text
    assert query.json()["answer"].strip().lower() == "v100"
    assert query.json()["sources"]

    search = client.post(
        f"/api/memory/sessions/{session_id}/search",
        json={"question": "K01", "mode": "lexical"},
    )
    assert search.status_code == 200
    assert search.json()["results"]

    adapt = client.post(
        f"/api/memory/sessions/{session_id}/adapt", json={"rounds": 2}
    )
    assert adapt.status_code == 200
    assert adapt.json()["total_updates"] >= 1

    save = client.post(f"/api/memory/sessions/{session_id}/save")
    assert save.status_code == 200
    restore = client.post(f"/api/memory/sessions/{session_id}/restore")
    assert restore.status_code == 200

    docs = client.get(f"/api/memory/sessions/{session_id}/documents")
    assert docs.status_code == 200
    doc_id = docs.json()["documents"][0]["doc_id"]
    removed = client.delete(f"/api/memory/sessions/{session_id}/documents/{doc_id}")
    assert removed.status_code == 200
    assert removed.json()["documents"] == []

    deleted = client.delete(f"/api/memory/sessions/{session_id}")
    assert deleted.status_code == 200
    missing = client.get(f"/api/memory/sessions/{session_id}")
    assert missing.status_code == 404


def test_training_job_and_checkpoint_selection(client):
    start = client.post(
        "/api/training/start",
        json={
            "task": "arith_add",
            "profile": "tiny",
            "epochs": 1,
            "train_size": 16,
            "val_size": 8,
            "test_size": 8,
            "batch_size": 8,
            "lr": 0.003,
            "warmup_steps": 1,
            "mode": "fixed_depth",
            "max_steps": 2,
            "min_complexity": 1,
            "max_complexity": 1,
            "checkpoint_name": "api-pytest-run",
        },
    )
    assert start.status_code == 200, start.text
    job_id = start.json()["job_id"]
    job = _wait_job(client, "/api/training/jobs", job_id)
    assert job["status"] == "completed", job
    result = job["result"]
    assert result["checkpoint_id"] == "api-pytest-run-final"
    assert result["parameters_total"] > 0
    assert result["history"]

    checkpoints = client.get("/api/checkpoints").json()["checkpoints"]
    ids = {c["checkpoint_id"] for c in checkpoints}
    assert "api-pytest-run-final" in ids

    selected = client.post("/api/checkpoints/api-pytest-run-final/select")
    assert selected.status_code == 200
    assert selected.json()["checkpoint_id"] == "api-pytest-run-final"

    generated = client.post(
        "/api/playground/generate",
        json={"prompt": "calc 1 + 2 =", "mode": "adaptive_hard", "max_new_tokens": 6},
    )
    assert generated.status_code == 200, generated.text
    assert generated.json()["checkpoint_id"] == "api-pytest-run-final"
    assert generated.json()["result"]["text"] is not None


def test_benchmark_job_and_report_listing(client):
    start = client.post(
        "/api/benchmarks/start",
        json={
            "suite": "memory_recall",
            "options": {
                "memory_records": 4,
                "adapt_rounds": 1,
                "device": "cpu",
            },
        },
    )
    assert start.status_code == 200, start.text
    job = _wait_job(client, "/api/benchmarks/jobs", start.json()["job_id"])
    assert job["status"] == "completed", job
    assert job["result"]["suite"] == "memory_recall"
    reports = client.get("/api/benchmarks/reports")
    assert reports.status_code == 200
    assert any(r["run_id"] == job["result"]["run_id"] for r in reports.json()["reports"])
    detail = client.get(f"/api/benchmarks/reports/{job['result']['run_id']}")
    assert detail.status_code == 200
    assert detail.json()["notes"]
    csv = client.get(f"/api/benchmarks/reports/{job['result']['run_id']}/csv")
    assert csv.status_code == 200
    assert "text/csv" in csv.headers["content-type"]
    assert "adapt_round" in csv.text


def test_api_key_guard(temp_app_cfg):
    cfg = temp_app_cfg.model_copy(deep=True)
    cfg.server.api_key = "secret"
    app = create_app(cfg)
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/config").status_code == 401
        assert (
            client.get("/api/config", headers={"X-API-Key": "secret"}).status_code
            == 200
        )
