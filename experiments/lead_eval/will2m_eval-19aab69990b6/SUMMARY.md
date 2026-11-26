# WILL-2M v3: verified lead evaluation

Source: `report.json` in this directory. Both checkpoints measured on the same
4-logical-core CPU with 2 Torch threads, 5 warmups and 50 timed runs. Test 3
uses batches of 16; Test 6 warm uses all 30 questions in one call.

## Architecture change over v2

- Right-to-left column scratchpad: raw ones sum -> raw tens sum (carry included)
  -> final answer copied from the emitted columns.
- Three generation calls per hard question at `fixed_depth=3`, so the recurrent
  block runs Loops 2-4.
- Replay remains 40% Easy / 40% Hard scratchpad / 20% Memory-conditioned.
- The memory path is unchanged: sequential answer slots + cosine-gated logit
  skip with a `recall` cue.

## Test 3 (100 held-out per stratum)

| Stratum | Checkpoint | Accuracy | Mean depth | p50 | p95 |
|---|---|---:|---:|---:|---:|
| Easy | 8M v1 | 80.0% | 1.268 | 153.4 ms | 327.6 ms |
| Easy | 2M v3 | **100.0%** | **1.000** | **42.0 ms** | **64.0 ms** |
| Hard | 8M v1 | 13.0% | 1.891 | 241.5 ms | 437.4 ms |
| Hard | 2M v3 | **100.0%** | **3.000** | 300.2 ms | 674.4 ms |

## Test 6 (30 held-out memory questions)

| Phase | Checkpoint | Accuracy | Mean depth | p50 | p95 |
|---|---|---:|---:|---:|---:|
| Cold | 8M v1 | 10.0% | 1.920 | 415.5 ms | 847.5 ms |
| Cold | 2M v3 | **100.0%** | 3.000 | **267.5 ms** | **568.3 ms** |
| Warm | 8M v1 | 3.3% | 1.417 | 413.8 ms | 869.4 ms |
| Warm | 2M v3 | **100.0%** | **1.000** | **124.7 ms** | **211.8 ms** |

Warm memory collapses depth from 3.000 to 1.000 and lowers p50 by 142.8 ms
relative to cold, while keeping accuracy at 100%.

Verification: `python scripts/verify_benchmarks.py` prints this run's report id
`will2m_eval-19aab69990b6` alongside the full architecture progression.
