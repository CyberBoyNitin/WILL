"""One-command verification of every published WILL benchmark claim.

Usage::

    python scripts/verify_benchmarks.py
    python scripts/verify_benchmarks.py --json

The script only reads already-produced report JSON files; it never invents or
recomputes numbers.  Report IDs are printed so each row can be inspected
directly on disk.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEAD = ROOT / "experiments" / "lead_eval"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_v3() -> Path | None:
    candidates = []
    for path in LEAD.glob("will2m_eval-*/report.json"):
        report = load(path)
        if report.get("comparison", {}).get("new") == "will-2m-synergy-v3":
            candidates.append((float(report.get("created_at", 0)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def row(label: str, report: dict[str, Any], tag: str) -> dict[str, Any]:
    result = report["results"][tag]
    strata = result["test3"]["strata"]
    test6 = result["test6"]
    return {
        "stage": label,
        "checkpoint_id": result["checkpoint_id"],
        "report_id": report["run_id"],
        "parameters": result.get("parameters"),
        "t3_easy_acc": strata["easy"].get("accuracy_pct"),
        "t3_easy_depth": strata["easy"].get("mean_depth"),
        "t3_easy_p50_ms": strata["easy"].get("latency_p50_ms"),
        "t3_hard_acc": strata["hard"].get("accuracy_pct"),
        "t3_hard_depth": strata["hard"].get("mean_depth"),
        "t3_hard_p50_ms": strata["hard"].get("latency_p50_ms"),
        "t6_cold_acc": test6["cold"].get("accuracy_pct"),
        "t6_warm_acc": test6["warm"].get("accuracy_pct"),
        "t6_cold_depth": test6["cold"].get("mean_depth"),
        "t6_warm_depth": test6["warm"].get("mean_depth"),
        "t6_depth_collapse": round(
            float(test6["warm"].get("mean_depth") or 0)
            - float(test6["cold"].get("mean_depth") or 0),
            3,
        ),
        "t6_warm_p50_ms": test6["warm"].get("latency_p50_ms"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    legacy = load(LEAD / "will8m_eval-cbb15aeef50f" / "report.json")
    rows.append(row("581K baseline", legacy, "old"))

    v2_report = load(LEAD / "will2m_eval-0b08da74f4e2" / "report.json")
    rows.append(row("8M v1", v2_report, "old"))
    rows.append(row("2M v2", v2_report, "new"))

    v3_path = find_v3()
    if v3_path is not None:
        rows.append(row("2M v3", load(v3_path), "new"))

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    header = (
        f"{'stage':<14}{'params':>10}{'T3 easy':>10}{'T3 hard':>10}"
        f"{'T6 cold':>10}{'T6 warm':>10}{'depth d':>10}{'report id':>26}"
    )
    print(header)
    print("-" * len(header))
    for item in rows:
        print(
            f"{item['stage']:<14}"
            f"{item['parameters']:>10,}"
            f"{item['t3_easy_acc']:>9.1f}%"
            f"{item['t3_hard_acc']:>9.1f}%"
            f"{item['t6_cold_acc']:>9.1f}%"
            f"{item['t6_warm_acc']:>9.1f}%"
            f"{item['t6_depth_collapse']:>10.2f}"
            f"{item['report_id']:>26}"
        )
    if v3_path is None:
        print("\n2M v3 report not found yet: run experiments/lead_eval/eval_will2m.py")
    print("\nAll numbers are read from the listed report.json files; none are estimated.")


if __name__ == "__main__":
    main()
