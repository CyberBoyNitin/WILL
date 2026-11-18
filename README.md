# WILL

**Adaptive recurrent computation with session-local fast-weight memory.**

WILL is a compact, runnable PyTorch research project. It combines a shared
recurrent reasoning block with per-token adaptive halting (ACT) and an optional
fast-weight memory that adapts inside an isolated session while the slow model
parameters stay frozen. It is not a chatbot wrapper and it does not use
simulated metrics: training, generation, retrieval and benchmark numbers in this
repository come from executed code.

The project includes:

- a byte-level tokenizer, causal context encoder, shared recurrent block,
  halting gate, adaptive controller and output head;
- differentiable ACT training with a ponder cost;
- hard inference-time early exit that actually skips halted token evaluations;
- TTT-linear, TTT-MLP and associative fast-weight memory;
- PDF/TXT/Markdown ingestion, BM25 + neural + hybrid retrieval, citations and
  explicit insufficient-information answers;
- a FastAPI backend with SQLite job storage and asynchronous training/benchmark
  jobs;
- a seven-page Next.js research console backed only by the live API;
- synthetic datasets, checkpoint evaluation, JSON/CSV benchmark reports and a
  pytest suite.

> **Scope and honesty.** The evidence here is from small synthetic tasks on CPU.
> It is not evidence of general language ability, production quality, or energy
> savings. MAC/FLOP values are analytic estimates. Hard ACT reduces token
> evaluations and estimated MACs, but on this CPU it is **slower in wall-clock
> time** than fixed depth because gather/scatter overhead dominates. No power
> measurement was made, so no energy claim is made.

---

## Architecture

```text
prompt bytes
    │
    ▼
byte tokenizer (260 ids: PAD/BOS/EOS/SEP + 256 bytes)
    │
    ▼
token + positional embeddings
    │
    ▼
causal context encoder (Transformer layers)
    │
    ▼
contextual state ───────────────┐
    │                           │
    ▼                           │
shared recurrent block ──► step embedding
    │        ▲                  │
    │        └── reused every step
    ▼
halting gate (per token, per step)
    │
    ├── fixed_depth: run exactly N steps
    ├── adaptive_soft: differentiable expected depth, all steps trained
    └── adaptive_hard: inference-only gather/scatter, halted tokens skipped
    │
    ▼
aggregator ──► optional fast-memory integration gate ──► output head
                                                        │
                                                        ▼
                                                   next byte
```

### Adaptive computation

The recurrent block is one shared module reused at every reasoning step. A
separate step embedding tells the block which iteration it is on. The halting
gate produces a per-token halting probability at each step, and the controller
turns those probabilities into a differentiable expected depth plus a ponder
cost.

- `fixed_depth` is the baseline: every token runs the same number of steps.
- `adaptive_soft` trains the halting distribution with all steps active. It is
  **not** a compute-saving inference mode; it is the differentiable ACT path.
- `adaptive_hard` is inference-only. It runs under `no_grad`, gathers only the
  tokens that have not halted, scatters their updated states back, and skips
  halted-token computation. This is the mode that reduces token evaluations.

`min_steps` prevents the gate from collapsing to one step. In the reported ACT
run, `min_steps=2`, `ponder_warmup_epochs=10` and `ponder_lambda=0.01` were used;
applying ponder pressure too early collapsed the gate to roughly 1.1 steps.

### Fast-weight memory

Memory is optional and session-local:

- `TTTLinear`, `TTTMLP` and `AssociativeMemory` implement bounded fast-weight
  state;
- a token-aware GRU `MemoryEncoder` and an integration gate inject memory into
  the contextual state;
- a `MemoryManager` owns UUID-isolated sessions, serialization, restore, reset,
  chunk bounds and adaptation rounds;
- adaptation updates **fast weights only**; the slow model is never written;
- retrieval can be lexical (BM25), neural (fast-weight prediction) or hybrid;
- answers carry source chunks and are marked `insufficient` below the retrieval
  threshold instead of inventing an answer.

The unified model also runs with memory disabled. Memory and ACT are separately
testable and are not required for each other.

---

## Measured results

All results below were produced on the machine described in
[Reproducibility](#reproducibility). The full reports are in
`experiments/benchmarks/`.

### Addition and adaptive depth

Checkpoint: `add-single-act-final` (581,765 parameters). Test split: 150
single-digit addition examples.

| Execution | Test accuracy | Mean depth | Generation token evals | Estimated generation MACs | Generation wall-clock |
|---|---:|---:|---:|---:|---:|
| fixed depth 1 | 67.3% | 1.00 | 5,486 | 0.986B | 0.489 s |
| fixed depth 2 | 94.0% | 2.00 | 10,972 | 1.884B | 0.532 s |
| fixed depth 3 | 99.3% | 3.00 | 16,458 | 2.783B | 0.632 s |
| fixed depth 4 | 99.3% | 4.00 | 21,944 | 3.682B | 0.855 s |
| `adaptive_soft` @ 4 | 99.3% | 2.137 | 21,944 | 3.682B | 1.140 s |
| `adaptive_hard` @ 4 | **99.3%** | **2.180** | **11,970** | **2.048B** | **0.973 s** |

Report: `adaptive_compute-1794893410`.

`adaptive_hard` matches the best fixed-depth accuracy while using **45.45% fewer
generation token evaluations** and about 44% fewer estimated generation MACs
than fixed depth 4. Its measured generation speedup is **0.878×**: 0.973 s
versus 0.855 s, so it is about 14% slower on this CPU. That overhead is the
honest result of Python gather/scatter bookkeeping around a tiny model.

The fixed-depth checkpoint `add-single-d4-final` (trained without ACT) reaches
96.0% validation accuracy. Its test accuracy by depth is:

| Depth | Test accuracy |
|---:|---:|
| 1 | 64.0% |
| 2 | 82.7% |
| 3 | 87.3% |
| 4 | 96.0% |

### Session-local memory

Synthetic key-value set with 20 records. Recall@1 is measured after adaptation
rounds; lexical BM25 is an index baseline, not a learned memory.

| Retrieval | Recall@1 |
|---|---:|
| neural fast weights, before adaptation | 40% |
| neural fast weights, 1 round | 80% |
| neural fast weights, 5 rounds | 95% |
| neural fast weights, 20 rounds | 95% |
| lexical BM25, 20 rounds | 100% |
| hybrid, 20 rounds | 95% |

At round 20 the neural state was 29,440 bytes, adaptation took 0.475 s, and the
slow parameter count remained 581,765. Report: `memory_recall-1794893574`.

This is a small synthetic recall benchmark. It shows that the fast-weight
adaptation path learns something real, not that the system is a general
retrieval model.

### Inference latency

Report: `inference-1794894784`. Prompt lengths 32 and 64, one repeat.

| Mode | Prompt tokens | p50 latency | Token evaluations |
|---|---:|---:|---:|
| fixed depth 2 | 32 | 31.795 ms | 64 |
| fixed depth 2 | 64 | 45.972 ms | 128 |
| `adaptive_hard` | 32 | 127.448 ms | 66 |
| `adaptive_hard` | 64 | 112.732 ms | 130 |

Hard mode evaluates roughly the same number of tokens as fixed depth 2 at these
short prompts but pays a large per-step control-flow cost. The token-evaluation
savings become visible at larger budgets; the wall-clock overhead does not
disappear.

### Ablation

Report: `ablation-1794894825`. A short combined run found 0.95 best fixed
accuracy at depth 2, 1.0 adaptive-hard accuracy at max depth 4, and 1.0
neural/lexical/hybrid recall on 4 records after 1 round. It is a smoke-scale
ablation, not a substitute for the longer runs above.

---

## Quickstart

### 1. Python environment

Requires Python 3.10+; this repository was exercised with Python 3.14.4 and
PyTorch 2.14.0+cpu.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

### 2. Train

The reported addition checkpoint can be reproduced with:

```powershell
python scripts/train.py --config configs/arith_add.yaml
```

A fixed-depth baseline:

```powershell
python scripts/train.py --task arith_add --min-complexity 1 --max-complexity 1 `
  --mode fixed_depth --max-steps 4 --epochs 30 --checkpoint-name add-d4
```

Checkpoints are written under `checkpoints/`; metadata is stored next to each
checkpoint. Training configs are saved inside the checkpoint so evaluation and
benchmark reports can be traced back to the exact run. The two reference
checkpoints used by the report (`add-single-act-final` and
`add-single-d4-final`) are tracked so a fresh clone can run the console
immediately; newly trained checkpoints are ignored by git.

### 3. Evaluate a checkpoint

```powershell
python -m scripts.evaluate_checkpoint checkpoints/add-single-act-final.pt `
  --split test --modes fixed_depth adaptive_soft adaptive_hard --depths 1 2 3 4 `
  --json-out experiments/add-single-act-eval.json
```

### 4. Run the API

```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the generated OpenAPI documentation.
The default checkpoint is selected from `checkpoints/` and can be changed with
`POST /api/checkpoints/{checkpoint_id}/select`.

### 5. Run the console

```powershell
cd frontend
copy .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`. For a production build:

```powershell
npm run build
npm run start
```

The console reads `NEXT_PUBLIC_API_URL` at build time. It does not contain
hard-coded result data; every number is fetched from the API.

### 6. Tests

```powershell
python -m pytest -q
```

Current suite: **37 tests passed**. It covers tokenization, model shapes and
masking, generation padding, memory isolation and adaptation, retrieval,
training/early stopping, benchmark reports, and API routes.

There is also a browser smoke test for the console:

```powershell
# with the API on :8000 and the console on :3000
cd frontend
npm run smoke
```

It launches the installed Chrome headless, checks all seven pages, queues and
finishes a one-epoch training job, creates and deletes a memory session, asks a
grounded question, runs generation and forward diagnostics, queues a real
memory-recall benchmark, and downloads its CSV. Override the browser with
`CHROME_PATH` if Chrome is elsewhere. The smoke test cleans up its temporary
checkpoints and sessions; its benchmark report remains as evidence.

CI is defined in `.github/workflows/ci.yml`: it installs CPU PyTorch, runs the
pytest suite, installs the frontend with `npm ci`, and builds the console.

---

## Configuration

Configuration is YAML, loaded by `backend.config.load_config`. Named profiles
provide architecture defaults, and explicit YAML values override them.

| Profile | Hidden | Heads | Context layers | Max steps | Memory dim |
|---|---:|---:|---:|---:|---:|
| `tiny` | 128 | 4 | 1 | 4 | 64 |
| `standard` | 256 | 4 | 2 | 8 | 128 |
| `experimental` | explicit | explicit | explicit | explicit | explicit |

Example files:

- `configs/arith_add.yaml` — the reported single-digit addition ACT recipe;
- `configs/standard.yaml` — larger GPU-oriented profile;
- `configs/experimental.yaml` — explicit non-profile architecture and mixture
  task.

Environment variables read by the backend:

| Variable | Purpose |
|---|---|
| `WILL_DEVICE` | `auto`, `cpu` or `cuda` |
| `WILL_DATA_DIR` | overrides data/upload/memory/database locations |
| `WILL_CHECKPOINT_DIR` | checkpoint directory |
| `WILL_EXPERIMENT_DIR` | benchmark/experiment directory |
| `WILL_API_KEY` | optional shared-secret API key |

Use `.env.example` as a reference and export those values in your process
environment. The backend does not load `.env` automatically. The frontend uses
`frontend/.env.local` and `NEXT_PUBLIC_API_URL` / `NEXT_PUBLIC_API_KEY`.

### API surface

| Area | Endpoints |
|---|---|
| System | `GET /api/health`, `/api/system`, `/api/config`, `/api/overview` |
| Checkpoints | `GET /api/checkpoints`, `/current`, `/{id}`, `/{id}/history`, `/{id}/evaluate`; `POST /{id}/select`; `DELETE /{id}` |
| Training | `POST /api/training/start`; `GET /defaults`, `/jobs`, `/jobs/{id}`, `/jobs/{id}/events`; `POST /jobs/{id}/cancel` |
| Benchmarks | `GET /api/benchmarks/suites`, `/jobs`, `/jobs/{id}`, `/jobs/{id}/events`, `/reports`, `/reports/{run_id}`, `/reports/{run_id}/csv`; `POST /start`, `/jobs/{id}/cancel` |
| Memory | session CRUD under `/api/memory/sessions`; ingest, ingest-text, documents, query, search, adapt, save, restore, reset |
| Datasets | `GET /api/datasets`, `/api/datasets/{task}/sample` |
| Playground | `GET /api/playground/presets`; `POST /tokenize`, `/generate`, `/forward` |
| Experiments | `GET /api/experiments`, `/api/experiments/{name}` |

Training and benchmark jobs run in a background worker and persist progress and
events in SQLite (`data/will.db` by default). The console polls those jobs; it
does not fabricate progress.

---

## Benchmarks

Benchmark suites:

| Suite | What it measures |
|---|---|
| `reasoning` | fixed-depth task accuracy/loss |
| `adaptive_compute` | depth, accuracy, token evaluations, analytic MAC estimates |
| `memory_recall` | lexical/neural/hybrid recall before and after adaptation |
| `inference` | latency, throughput, token evaluations and estimated MACs |
| `ablation` | execution-mode and memory-subsystem comparison |

Start one through the console or the API:

```powershell
$body = @{
  suite = "adaptive_compute"
  checkpoint_id = "add-single-act-final"
  limit = 150
} | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/benchmarks/start `
  -ContentType "application/json" -Body $body
```

Reports are written as JSON and CSV under:

```text
experiments/benchmarks/<run_id>/report.json
experiments/benchmarks/<run_id>/summary.csv
```

Every report includes the resolved options, checkpoint, hardware/software
metadata, measured rows, duration and notes. MAC/FLOP values are explicitly
labelled as analytic estimates. No benchmark measures power or energy.

---

## Project layout

```text
backend/
  engine/       tokenizer, embeddings, context encoder, recurrent block,
                halting gate, controller, output head, unified model, metrics
  memory/       fast weights, memory encoder/manager, BM25, ingestion, retrieval
  training/     datasets, losses, trainer, checkpoints
  benchmarks/   suite runners and JSON/CSV report writing
  server/       FastAPI app, routes, SQLite store, jobs, schemas, services
frontend/
  app/          seven console pages (overview, model, training, benchmarks,
                memory, playground, experiments)
  components/   shared UI and custom SVG charts
  lib/          API client, hooks and types
  scripts/      browser smoke test
tests/          pytest suite
scripts/        train and evaluate command-line entry points
configs/        YAML configurations
experiments/    training/evaluation JSON and benchmark reports
checkpoints/    generated .pt checkpoints and metadata
```

The seven console pages are:

1. **Overview** — system state, default checkpoint, parameter count, recent jobs
   and latest report.
2. **Model** — architecture, parameter mass, execution modes and data flow.
3. **Training** — start/cancel training jobs, inspect loss/accuracy history.
4. **Benchmarks** — queue suites, watch progress, browse reports and download CSV.
5. **Memory** — create sessions, ingest files/text, adapt, query with citations,
   inspect retrieval scores and session state.
6. **Playground** — generate, run forward diagnostics, inspect token-level depth
   and entropy, and tokenize bytes.
7. **Experiments** — checkpoint registry, evaluation, training history and raw
   artifact browser.

---

## Reproducibility

The reported runs used:

- Windows 10, Python 3.14.4;
- PyTorch 2.14.0+cpu, CUDA unavailable;
- 4 logical CPUs, 2 PyTorch threads;
- seed 1337 for training datasets;
- tiny profile: 581,765 parameters.

Key report IDs:

| Report | Run ID |
|---|---|
| ACT accuracy/compute | `adaptive_compute-1794893410` |
| Fixed-depth comparison | `adaptive_compute-1794893481` |
| Reasoning suite | `reasoning-1794987234` |
| Memory recall | `memory_recall-1794893574` |
| Inference latency | `inference-1794894784` |
| Ablation | `ablation-1794894825` |

The JSON reports are the source of truth for the tables above. Regenerate them
rather than copying numbers between machines.

---

## Limitations

- **Synthetic tasks only.** The checkpoints are trained on generated addition,
  multi-hop, sequence and long-context tasks. They are not general-purpose
  language models.
- **Small evaluation sets.** The main adaptive-compute test split has 150
  examples; the inference report uses one repeat. Treat the numbers as
  directional, not statistically strong.
- **Hard ACT is not a CPU speedup.** It reduces token evaluations and estimated
  MACs, but gather/scatter overhead made it 0.878× the wall-clock speed of fixed
  depth 4 in the reported run.
- **No energy claim.** There is no power meter in the loop. Estimated MACs are
  not energy.
- **Adaptive soft does not skip work.** It is the differentiable training path;
  only `adaptive_hard` performs hard token skipping.
- **Memory recall is task-specific.** Neural recall improves from 40% to 95% on
  the synthetic key-value benchmark. Lexical BM25 is stronger there; hybrid is
  not automatically best.
- **Old checkpoints are not comparable.** Early checkpoints predate the
  answer-token/EOS-only loss masking. Only checkpoints trained with the current
  loss should be compared.
- **PDF ingestion is text extraction, not OCR.** Scanned image-only PDFs may
  yield no text.
- **Single shared API key.** `WILL_API_KEY` is authentication for a trusted local
  deployment, not a multi-user authorization system.

---

## License

MIT. See `LICENSE`.
