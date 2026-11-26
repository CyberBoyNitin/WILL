# WILL — Adaptive Fast-Weight Transformer

**Adaptive recurrent computation with sequential slot fast-weight memory.**

WILL is a compact PyTorch research architecture that couples:

1. **Active-Batch Compaction** — per-token adaptive halting with gather/scatter so halted tokens stop consuming recurrent evaluations;
2. **Sequential Slot Fast-Weight Memory** — session-local TTT fast weights that store one token embedding per answer position instead of one pooled vector;
3. **Cosine-Gated Logit Skip** — a high-confidence retrieval path that adds retrieved token similarities directly to the LM logits;
4. **Right-to-Left Carry Scratchpad** — for two-digit arithmetic, the model emits raw column sums from right to left, then splits each sum into digit + carry before assembling the answer.

Everything in this repository is executed code: checkpoints, held-out benchmarks, API responses and telemetry. No simulated accuracy, no power measurements, no CUDA claims.

---

## One-command verification

```bash
python scripts/verify_benchmarks.py
python scripts/smoke_api.py          # with the API running on :8000
python -m pytest -q
```

`verify_benchmarks.py` reads only the committed `report.json` files and prints the report IDs behind every number in the table below.

---

## Verified benchmark progression

| Stage | Checkpoint | Active params | Test 3 Easy | Test 3 Hard | Test 6 Warm | Report ID |
|---|---|---:|---:|---:|---:|---|
| 581K baseline | `add-mixed-70-30-v2-final` | 610,759 | 100.0% @ depth 2.107 | 11.0% @ depth 2.139 | 13.3% | `will8m_eval-cbb15aeef50f` |
| 8M v1 | `will-8m-synergy-v1` | 8,044,904 | 80.0% @ depth 1.268 | 13.0% @ depth 1.891 | 3.3% | `will2m_eval-0b08da74f4e2` |
| 2M v2 | `will-2m-synergy-v2` | 2,014,080 | **100.0% @ depth 1.000** | 24.0% @ depth 4.000 | **100.0% @ depth 1.000** | `will2m_eval-0b08da74f4e2` |
| 2M v3 | `will-2m-synergy-v3` | 2,014,080 | **100.0% @ depth 1.000** | **100.0% @ depth 3.000** | **100.0% @ depth 1.000** | `will2m_eval-19aab69990b6` |

The v2/v3 rows are measured with the same harness: 5 warmups, 50 timed runs, CPU / 2 threads, Test 3 batch 16, Test 6 all 30 questions in one call. The 581K row uses the earlier v4 lead-eval harness and is included as the historical starting point; its report ID is listed so the protocol difference is auditable.

---

## Architecture

### Active-Batch Compaction

The context encoder builds a causal representation once. A shared recurrent block is then applied for up to `max_steps` iterations, with a learned step embedding. A per-token halting gate emits a stopping probability at each step:

- `adaptive_soft` trains on the differentiable expected depth with a ponder penalty;
- `adaptive_hard` is inference-only — halted positions are removed from the active batch via gather/scatter, so `mean_depth` reflects real skipped evaluations;
- `fixed_depth` runs exactly N steps and is used for the difficulty-routed arithmetic benchmarks.

This is the mechanism behind the measured `4.000 → 1.000` depth collapse when memory is consulted.

### Sequential Slot Fast-Weight Memory

The old pooled-memory path compressed a whole answer into one vector; multi-digit answers had to be guessed from that single point. The v2+ writer stores one key/value pair per answer-token position:

```
key[j]   = normalize(pooled_record_key * slot_sign[j] + slot_address[j])
value[j] = token_embedding(answer_token[j])
```

Slot-specific sign flips keep the four key sets linearly independent, and a fixed random lexical code prevents short, similar prompts from collapsing onto the same record key. The ridge solve (`1e-6`) reproduces stored token embeddings with cosine ≈ 1.0. The writer and reader share the same pooled key, padding positions and slot signs, so retrieval is exact rather than approximate.

### Cosine-Gated Logit Skip

When `memory_logit_skip=True`, each slot value is normalised and scored against the token-embedding matrix:

```
logits += alpha * 1[max_slot_cosine > 0.85] * slot_cosine_scores
```

`alpha` is fixed at 20.0 for new configs. Legacy checkpoints default the flag to `False` and a zero alpha, so the 8M baseline behaviour is unchanged. The skip is applied at generation step *j* from memory slot *j*, which is why a multi-digit answer can be emitted directly at depth 1.

### Right-to-Left Carry Scratchpad

For Complexity-2 questions, v3 uses a three-stage right-to-left chain. Each stage is a short generation whose first token is the supervised value:

```
calc 47 + 38 =                  -> 15    raw ones column: 7 + 8
calc 47 + 38 = ones 15          -> 8     raw tens column: 4 + 3 + carry(15)
calc 47 + 38 = ones 15 tens 8   -> 85    tens sum ++ last ones digit
```

The carry is resolved implicitly by the tens-sum stage (`15` implies a carry of 1), which avoids the mod-10 digit extraction that stalled earlier scratchpad variants. Reading the final answer is then a copy of the last character of the emitted ones sum. Python never computes the carries at inference: every stage consumes only the model's own emitted text. Operand pairs are sampled fresh in every training batch, so this is rule learning rather than memorising the 1,600-example training file.

---

## Install and run

```bash
pip install -e ".[server,test]"

will-serve --host 127.0.0.1 --port 8000
# or: python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

### Python SDK

```python
from will_engine import WILLModel, WILLConfig, WILLForCausalLM

cfg = WILLConfig(hidden_dim=168, memory_dim=168, memory_slots=4,
                 memory_logit_skip=True, memory_logit_scale=20.0)
model = WILLModel(cfg)

engine = WILLForCausalLM("will-2m-synergy-v2")
print(engine.generate("calc 2 + 3 =", max_new_tokens=6,
                      mode="fixed_depth", max_steps=1).text)
```

### OpenAI-compatible endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Streaming and non-streaming chat completions with `will` diagnostics |
| `POST /v1/memory/absorb` | Ingest text into a session-local fast-weight state |
| `POST /v1/memory/query` | Lexical / neural / hybrid memory query |
| `GET /v1/telemetry` | Real generation counts, FLOP estimates and latency |
| `GET /v1/models` | Available checkpoints |

`X-API-Key` is required only when `server.api_key` is configured. All legacy `/api` routes keep their original schemas.

---

## Reproduce

```bash
# v2 checkpoint and its 40/40/20 replay + polish
python scripts/train_will2m.py --threads 4
python scripts/replay_will2m.py --threads 4
python scripts/polish_will2m.py --threads 4

# v3 right-to-left carry curriculum (starts from v2)
python scripts/train_will2m_v3.py --threads 4 --out will-2m-synergy-v3-base
# answer-stage polish, then depth-3 adaptation to the requested 3-4 loop band
python scripts/polish_will2m_v3.py --threads 4 --steps 300 --hard-weight 0.5 --lr 1e-4 --init will-2m-synergy-v3-base --out will-2m-synergy-v3
python scripts/train_will2m_v3.py --threads 4 --init will-2m-synergy-v3 --out will-2m-synergy-v3-d3 --depth 3 --stage-steps 300 --replay-steps 100 --stage-lr 2e-4 --replay-lr 1e-4
# promote the depth-3 checkpoint to the published id (see scripts/promote_will2m_v3.py)

# Exact Test 3 / Test 6 measurements (5 warmups, 50 timed runs)
python experiments/lead_eval/eval_will2m.py

# Verify every published number
python scripts/verify_benchmarks.py
```

Checkpoints are gitignored; the scripts above regenerate them deterministically from seed 1337.

---

## Honest limitations

- Evidence is on synthetic byte-level arithmetic and synthetic memory replay, not general language ability.
- Test 3 Hard reached 100% only after the right-to-left column chain and a dedicated depth-3 adaptation; the earlier direct-answer 2M checkpoint was 24%.
- p50 under 150 ms is met for Easy (42 ms) and warm memory (125 ms); the chained hard decoder is ~300 ms on this 4-core CPU.
- `torch.compile` could not be used (`cl` compiler unavailable); no GPU/CUDA or energy result is claimed.
- MAC/FLOP figures, where present, are analytic estimates, not power measurements.

MIT licensed. See `LICENSE`.
