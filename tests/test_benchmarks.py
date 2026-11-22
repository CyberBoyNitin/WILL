"""Benchmark runner and report tests."""
from __future__ import annotations

import json
from pathlib import Path

from backend.benchmarks.runner import (
    list_reports,
    load_report,
    run_benchmark_suite,
)


def test_memory_recall_benchmark_is_real_and_persisted(temp_app_cfg):
    report = run_benchmark_suite(
        temp_app_cfg,
        "memory_recall",
        {
            "memory_records": 5,
            "adapt_rounds": 2,
            "device": "cpu",
        },
    )
    assert report["format"] == "will-benchmark-v1"
    assert report["suite"] == "memory_recall"
    assert report["results"]
    assert report["summary"]["records"] == 5
    assert report["summary"]["adapt_rounds"] == 2
    assert 0.0 <= report["summary"]["neural_recall_at_1"] <= 1.0
    assert report["artifacts"]["json"]
    assert Path(report["artifacts"]["json"]).exists()
    assert Path(report["artifacts"]["csv"]).exists()
    saved = json.loads(Path(report["artifacts"]["json"]).read_text(encoding="utf-8"))
    assert saved["run_id"] == report["run_id"]
    assert saved["notes"]


def test_adaptive_compute_benchmark_reports_measured_fields(temp_app_cfg):
    report = run_benchmark_suite(
        temp_app_cfg,
        "adaptive_compute",
        {
            "limit": 6,
            "depths": [1, 2],
            "modes": ["adaptive_hard"],
            "batch_size": 6,
            "max_new_tokens": 4,
            "max_examples": 2,
            "device": "cpu",
        },
    )
    assert report["suite"] == "adaptive_compute"
    assert len(report["results"]) == 3
    for row in report["results"]:
        assert "accuracy" in row
        assert "loss" in row
        assert "generation_mean_depth" in row
        assert "generation_token_evaluations" in row
        assert "estimated_macs" in row
        assert row["generation_token_evaluations"] > 0
    fixed_baseline = [r for r in report["results"] if r["mode"] == "fixed_depth"][-1]
    adaptive = [r for r in report["results"] if r["mode"] == "adaptive_hard"][0]
    assert fixed_baseline["generation_compute_vs_max_fixed_pct"] == 0.0
    assert -100.0 <= adaptive["generation_compute_vs_max_fixed_pct"] <= 100.0


def test_continual_benchmark_proves_retention_and_save_load(temp_app_cfg):
    report = run_benchmark_suite(
        temp_app_cfg,
        "continual",
        {
            "continual_domains": 2,
            "continual_facts_per_domain": 5,
            "device": "cpu",
        },
    )
    assert report["suite"] == "continual"
    assert report["summary"]["facts_total"] == 10
    assert report["summary"]["neural_recall_at_1"] == 1.0
    assert report["summary"]["hybrid_recall_at_1"] == 1.0
    assert report["summary"]["max_neural_forgetting_pct"] == 0.0
    assert report["summary"]["catastrophic_forgetting_ok"] is True
    assert report["summary"]["slow_weights_unchanged"] is True
    assert report["summary"]["save_load_exact"] is True
    assert report["summary"]["state_version"] == 2
    assert len(report["results"]) == 2
    assert report["save_load"]["exact"] is True


def test_report_listing_and_loading(temp_app_cfg):
    report = run_benchmark_suite(
        temp_app_cfg,
        "reasoning",
        {
            "limit": 4,
            "depths": [1],
            "batch_size": 4,
            "max_new_tokens": 3,
            "max_examples": 1,
            "device": "cpu",
        },
    )
    listed = list_reports(temp_app_cfg.server.experiment_dir)
    assert any(item["run_id"] == report["run_id"] for item in listed)
    loaded = load_report(temp_app_cfg.server.experiment_dir, report["run_id"])
    assert loaded["run_id"] == report["run_id"]
