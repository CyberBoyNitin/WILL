# WILL-2M vs WILL-8M: measured comparison

Source: `report.json` in this directory. Both checkpoints were measured on the
same 4-logical-core CPU with `torch` configured for 2 threads, 5 warmup passes
and 50 timed runs. Test 3 uses generation batches of 16; Test 6 uses all 30
questions in one call.

## Architecture / checkpoint

| | will-8m-synergy-v1 | will-2m-synergy-v2 |
|---|---:|---:|
| Active parameters | 8,044,904 | 2,014,080 |
| Fast-state bytes | 66,048 | 113,568 |
| Memory path | pooled ridge retrieval | 4 sequential answer-slot values |
| Logit skip | disabled | enabled, cosine-gated at 0.85 |
| Test 3 hard depth | adaptive (mean 1.89) | fixed 4 loops |

## Test 3 (100 held-out per stratum)

| Stratum | Checkpoint | Accuracy | Mean depth | p50 | p95 |
|---|---|---:|---:|---:|---:|
| Easy | 8M | 80.0% | 1.268 | 153.4 ms | 327.6 ms |
| Easy | 2M | **100.0%** | 1.000 | **43.1 ms** | **86.3 ms** |
| Hard | 8M | 13.0% | 1.891 | 241.5 ms | 437.4 ms |
| Hard | 2M | **24.0%** | 4.000 | 276.2 ms | 553.8 ms |

## Test 6 (30 held-out memory questions)

| Phase | Checkpoint | Accuracy | Mean depth | p50 | p95 |
|---|---|---:|---:|---:|---:|
| Cold | 8M | 10.0% | 1.920 | 415.5 ms | 847.5 ms |
| Cold | 2M | **33.3%** | 4.000 | **241.5 ms** | **366.4 ms** |
| Warm | 8M | 3.3% | 1.417 | 413.8 ms | 869.4 ms |
| Warm | 2M | **100.0%** | **1.000** | **254.8 ms** | **413.4 ms** |

Warm memory reduces depth from 4 to 1 (`-3.0`) while raising accuracy from
33.3% to 100%. Relative to the 8M warm path it is +96.7 accuracy points,
159 ms lower p50, and a much smaller fast-state read cost per generated token.

## Notes

- Memory is written with `backend.memory.slot_memory.write_slot_memory` at
  ridge 1e-6; writer and reader share the exact pooled key, position mapping
  and slot signs, so retrieved slot values have cosine ~1.0 with the stored
  token embeddings. The `recall` cue separates memory-mode prompts from the
  arithmetic prompts so replay does not overwrite the easy mapping.
- A legacy `memory_logit_skip=False` checkpoint is unchanged: the new slot
  modules are present but inactive, and `mem_to_vocab` is only tied when
  `memory_dim == hidden_dim`.
- The explicit `ones,carry1,tens,carry2,answer` scratchpad was trained (and
  chained variants were measured), but direct 4-loop decoding generalised
  better on held-out two-digit questions, so the shipped checkpoint uses the
  direct path for Test 3. The 24% hard accuracy is a real improvement over
  13% but is far from solved.
- p50 under 150 ms is met for Easy (43 ms); hard/memory paths remain above
  150 ms on this 4-core CPU. No GPU, CUDA, or compiled kernel was available,
  so no energy claim is made.
- Training is CPU-only. Full reproduce chain: `python scripts/train_will2m.py`,
  `python scripts/replay_will2m.py`, `python scripts/polish_will2m.py`.
