"""Promote a validated v3 candidate checkpoint to `will-2m-synergy-v3`.

Usage::

    python scripts/promote_will2m_v3.py --src will-2m-synergy-v3-d3

The functional checkpoint files are gitignored; this only rewires their ids.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument("--src", default="will-2m-synergy-v3-d3")
p.add_argument("--dst", default="will-2m-synergy-v3")
p.add_argument("--checkpoint-dir", default="checkpoints")
args = p.parse_args()

root = Path(args.checkpoint_dir)
src = root / f"{args.src}.pt"
dst = root / f"{args.dst}.pt"
payload = torch.load(src, map_location="cpu", weights_only=False)
payload["checkpoint_id"] = args.dst
payload["training_config"]["checkpoint_name"] = args.dst
if isinstance(payload.get("metrics"), dict):
    payload["metrics"]["run_id"] = args.dst
if isinstance(payload.get("extra"), dict):
    payload["extra"]["run_id"] = args.dst
torch.save(payload, dst)
meta = {
    key: value
    for key, value in payload.items()
    if key not in ("model_state", "optimizer_state", "history")
}
meta["path"] = str(dst)
meta["size_bytes"] = dst.stat().st_size
(root / f"{args.dst}.json").write_text(
    json.dumps(meta, indent=2), encoding="utf-8"
)
print(f"promoted {src} -> {dst} ({dst.stat().st_size} bytes)")
