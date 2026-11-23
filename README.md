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
- an installable `will_engine` SDK and OpenAI-compatible `/v1` routes;
- a seven-page Next.js research console backed only by the live API;
- synthetic datasets, checkpoint evaluation, JSON/CSV benchmark reports and a
  pytest suite.

> **Scope and honesty.** The evidence here is from small synthetic tasks on CPU.
> It is not evidence of general language ability, production quality, or energy
> savings. MAC/FLOP values are analytic estimates. With incremental KV caching,
> `adaptive_hard@3` is faster than fixed depth 4 in the reported runs, but the
> margin depends on prompt length, machine load and CPU affinity; it is not a
> universal speedup guarantee. CUDA was unavailable, so no CUDA or GPU
> mixed-precision speedup is claimed. No power measurement was made, so no
> energy claim is made.

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
`experiments/benchmarks/`. Every row is real model execution; MAC/FLOP values
are analytic estimates and are labelled as such.

### Phase 1: cached generation and adaptive compute

Checkpoint: `add-single-act-final` (581,765 parameters), 150 held-out
single-digit addition examples. Generation times are medians of five repeats.

| Execution | Test accuracy | Mean depth | Generation token evals | Estimated generation MACs | Generation wall-clock |
|---|---:|---:|---:|---:|---:|
| fixed depth 1 | 67.3% | 1.00 | 2,321 | 0.467B | 0.440 s |
| fixed depth 2 | 94.0% | 2.00 | 4,642 | 0.847B | 0.637 s |
| fixed depth 3 | 99.3% | 3.00 | 6,963 | 1.228B | 1.046 s |
| fixed depth 4 | 99.3% | 4.00 | 9,284 | 1.608B | 1.308 s |
| `adaptive_soft` @ 3 | 99.3% | 2.074 | 6,963 | 1.228B | 0.940 s |
| `adaptive_hard` @ 3 | **99.3%** | **2.088** | **4,860** | **0.883B** | **0.628 s** |

Report: `adaptive_compute-1795066048`.

The three-step cap preserves the checkpoint's best accuracy. `adaptive_hard@3`
uses **47.65% fewer generation token evaluations** than fixed depth 4 and runs
**2.08× faster** in this cached, warmed run (0.628 s vs 1.308 s). This is the
opposite direction from the earlier uncached benchmark: incremental KV caching
removed the repeated full-prefix work that used to dominate the hard path.
`adaptive_soft` is still a training/analysis path and is not claimed to save
inference compute.

#### Latency at short prompts

Report: `inference-1795166202`, 50 repeats per row, high process priority and
CPU affinity `0xE` (logical CPUs 1–3). The run uses 8 new tokens and the
`add-single-act-final` checkpoint.

| Mode | Prompt tokens | p50 latency | p95 latency | Token evaluations |
|---|---:|---:|---:|---:|
| fixed depth 4 | 32 | 16.482 ms | 42.618 ms | 128 |
| fixed depth 4 | 64 | 15.974 ms | 25.853 ms | 256 |
| fixed depth 4 | 128 | 20.755 ms | 34.516 ms | 512 |
| `adaptive_hard` @ 3 | 32 | **12.427 ms** | 23.435 ms | **65** |
| `adaptive_hard` @ 3 | 64 | **14.128 ms** | 21.865 ms | **129** |
| `adaptive_hard` @ 3 | 128 | **17.491 ms** | 67.833 ms | **259** |

The short-prompt p50 is **12.427 ms**, below the 45 ms target, and adaptive hard
is faster than fixed depth 4 at all three lengths in this controlled run. The
p95 at 128 tokens is noisy (67.8 ms), so the p50 is the stable claim. Without
the priority/affinity control the same code produced much larger and sometimes
reversed differences; treat sub-millisecond comparisons as machine-dependent.

#### `torch.compile` probe

The inference report also attempts `torch.compile(model,
mode="reduce-overhead")` on the full-sequence forward, compares compiled and
eager logits, and records the result. On this Windows machine the probe fails
with `InductorError: InvalidCxxCompiler: Compiler: cl is not found` and falls
back to eager. The status is stored under `summary.compilation`; the KV-cached
generation path deliberately stays eager. No compiled latency is claimed.

### Phase 2: sequential continual learning

Three domains × 20 facts, learned sequentially with replay consolidation.
The slow weights are frozen and hashed before/after.

| Metric | Result |
|---|---:|
| Facts stored | 60 |
| Neural recall@1 | 100% |
| Lexical recall@1 | 100% |
| Hybrid recall@1 | 100% |
| Max neural forgetting | 0.0% |
| Max hybrid forgetting | 0.0% |
| Slow-weight hash unchanged | yes |
| Save/load exact | yes |
| Fast-state version | 2 |
| Consolidations | 3 |
| Fast state size | 55,040 bytes |
| Fast / slow parameters | 4,160 / 581,765 |

Report: `continual-1795067513`. The versioned state and session payloads load
version 1 sessions backwards-compatibly; replay uses a closed-form ridge solve
with an optional norm cap, and the slow model is never written.

### Phase 3: mixed difficulty and parameter scaling

#### Mixed 70/30 difficulty

Checkpoint `add-mixed-70-30-v2-best` (581,765 parameters), trained on a
70% one-digit / 30% two-digit addition mixture. Test split: 500 examples,
exactly 350 trivial and 150 hard.

| Execution | Overall accuracy | Trivial accuracy | Hard accuracy | Generation wall-clock |
|---|---:|---:|---:|---:|
| fixed depth 4 | 73.2% | 100.0% | 10.7% | 1.671 s |
| `adaptive_soft` @ 3 | **73.8%** | 100.0% | **12.7%** | 1.443 s |
| `adaptive_hard` @ 3 | 73.6% | 100.0% | 12.0% | 1.849 s |

Report: `mixed_difficulty-1795166057`. The composition check passes exactly
(`easy_ratio_observed = 0.7`). The model solves the trivial stratum perfectly
and the hard stratum poorly; the overall number is dominated by the 70% trivial
share and is reported next to the per-stratum breakdown rather than instead of
it.

#### 15M / 50M / 125M presets

Each preset is instantiated, its exact parameter count is reported, and real
forward/backward steps are executed. Shape: batch 1, padded sequence length 16,
2 recurrent steps, 1 warmup step, 2 measured steps. No CUDA device was
available.

| Preset | Target | Exact parameters | Delta | Hidden | Layers | fp32 step | bf16 step | fp16 step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `15m` | 15,000,000 | 15,237,365 | +1.58% | 624 | 2 | 0.114 s | 3.175 s | 3.262 s |
| `50m` | 50,000,000 | 50,107,013 | +0.21% | 1,152 | 2 | 0.595 s | 14.030 s | 16.024 s |
| `125m` | 125,000,000 | 123,920,165 | −0.86% | 1,824 | 2 | 1.088 s | 37.744 s | 51.339 s |

Report: `scaling-1795164427`. All nine precision rows produced finite losses
and gradients. The CPU bf16/fp16 steps are **slower**, not faster, because this
CPU has no native bf16 path and PyTorch falls back to emulation. That is the
honest measurement; it is not evidence against GPU mixed precision. CUDA is
recorded as unavailable (`cuda_available: false`), and no CUDA speedup is
claimed. The 125M preset was validated for step execution, not trained to
convergence.

### Session-local memory

Synthetic key-value set with 20 records and 20 adaptation rounds. Recall@1 is
measured after adaptation; lexical BM25 is an index baseline, not a learned
memory.

| Retrieval | Recall@1 |
|---|---:|
| neural fast weights | 100% |
| lexical BM25 | 100% |
| hybrid | 100% |

The neural state is 29,440 bytes and the slow parameter count remains 581,765.
Report: `memory_recall-1795238098`. This run includes the hybrid-scoring fix
that aligns hybrid ranking with the stored-row order used by neural recall.

### Phase 4: SDK and OpenAI-compatible API

`will_engine.WILLForCausalLM` loads a checkpoint, exposes synchronous and
streaming generation, session-local memory absorption/query, and process
telemetry. The FastAPI app mounts `/v1/chat/completions` (including SSE
streaming), `/v1/memory/absorb`, `/v1/memory/query`, `/v1/telemetry` and
`/v1/models` without changing any existing `/api` route. The pytest suite
exercises the SDK and all new HTTP endpoints, including streaming chunks,
memory-session compatibility, API-key protection and telemetry counters.

A live `uvicorn backend.app:app` smoke test on `127.0.0.1:8123` returned the
checkpoint registry, a non-streaming completion with `finish_reason="stop"`,
an SSE stream ending in `data: [DONE]`, and telemetry with a non-zero streaming
counter and `energy_measured=false`.

---

## Quickstart

### 1. Python environment

Requires Python 3.11+; this repository was exercised with Python 3.14.4 and
PyTorch 2.14.0+cpu.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

For the installable SDK and the `will-serve` command, use the package extras:

```powershell
pip install -e ".[server,test]"
```

The wheel exposes `will_engine` and the `backend` package together; the server
extras add FastAPI, Uvicorn, PDF ingestion and multipart uploads. Installed
deployments can set `WILL_HOME` (or `WILL_CHECKPOINT_DIR` and `WILL_DATA_DIR`)
so checkpoints and session state live outside site-packages. Without those
variables, writable state defaults to the current working directory rather than
the package directory.

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

The mixed 70/30 checkpoint used in the Phase 3 table:

```powershell
python scripts/train.py --task arith_add --min-complexity 1 --max-complexity 2 `
  --mixed-easy-ratio 0.7 --mode adaptive_soft --min-steps 2 --max-steps 4 `
  --epochs 15 --train-size 5000 --val-size 500 --test-size 500 `
  --batch-size 32 --lr 5e-4 --ponder-lambda 0.01 --ponder-warmup-epochs 5 `
  --checkpoint-name add-mixed-70-30-v2
```

The trainer writes `add-mixed-70-30-v2-best` at the best validation loss; the
Phase 3 report uses that checkpoint. Train one of the parameter-scaled presets
with `--scaling-preset 15m|50m|125m`.
The presets are architecture definitions, not simulated parameter counts:
`scaling_model_config()` returns a normal `ModelConfig`, and the scaling
benchmark instantiates it and reports the exact count.

Mixed precision and optional compilation are explicit training flags:

```powershell
python scripts/train.py --config configs/arith_add.yaml `
  --precision bf16 --torch-compile --compile-mode reduce-overhead
```

`precision` is recorded in the checkpoint metrics. `auto` selects bf16 only on
CUDA devices that report bf16 support; explicit `bf16`/`fp16` requests are
honoured or rejected, never silently downgraded. `torch.compile` runs a
numerical probe against eager and falls back with a machine-readable reason
when the compiler is unavailable or the logits differ.

Checkpoints are written under `checkpoints/`; metadata is stored next to each
checkpoint. Training configs are saved inside the checkpoint so evaluation and
benchmark reports can be traced back to the exact run. The two reference
checkpoints used by the Phase 1/2 tables (`add-single-act-final` and
`add-single-d4-final`) are tracked so a fresh clone can run the console
immediately; newly trained checkpoints, including the Phase 3 mixed checkpoint,
are ignored by git and can be reproduced with the commands above.

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
`POST /api/checkpoints/{checkpoint_id}/select`. After `pip install -e .`, the
same server starts with `will-serve --host 127.0.0.1 --port 8000`.

### 5. Use the SDK

```python
from will_engine import WILLForCausalLM

engine = WILLForCausalLM("add-single-act-final", device="cpu")
result = engine.generate("calc 2 + 3 =", max_new_tokens=4)
print(result.text, result.finish_reason, result.usage)

for event in engine.stream("calc 7 + 1 =", max_new_tokens=4):
    if event["type"] == "token":
        print(event["text"], end="", flush=True)

absorbed = engine.absorb(
    "The east vault access code is 4815. The backup code is 9021.",
    name="demo",
)
print(absorbed["session_id"], absorbed["chunks_added"])
print(engine.query(absorbed["session_id"], "east vault code?")["answer"])
print(engine.telemetry()["generation"])
```

`WILLForCausalLM` loads the checkpoint, byte tokenizer, fast-memory manager and
telemetry. It accepts `checkpoint_dir`, `memory_dir`, `config` and `device`.
`generate` returns a `GenerationResult`; `stream` yields `start`, `token`,
`done` and `error` events. Memory sessions are checkpoint-scoped: using a
session created by a different checkpoint is rejected instead of silently
mixing incompatible fast weights.

### 6. OpenAI-compatible API

```powershell
$body = @{
  model = "add-single-act-final"
  messages = @(@{ role = "user"; content = "calc 2 + 3 =" })
  max_tokens = 4
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/v1/chat/completions `
  -ContentType "application/json" -Body $body
```

Set `stream = $true` to receive server-sent events ending in `data: [DONE]`.
The response includes the standard `id`, `object`, `choices` and `usage` fields
plus a `will` object with checkpoint, mode, mean depth, token evaluations,
estimated FLOPs and latency. `/v1/models` lists the checkpoint registry;
`/v1/memory/absorb` ingests text; `/v1/memory/query` answers with citations;
`/v1/telemetry` returns process-wide counters and explicitly reports
`energy_measured: false`.

WILL is a task model, not an instruction-tuned chat model. The default
`chat_template: "plain"` passes message text through unchanged; it does not
invent a chat format the checkpoint never saw.

### 7. Run the console

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

### 8. Tests

```powershell
python -m pytest -q
```

Current suite: **54 tests passed**. It covers tokenization, model shapes and
masking, generation padding, KV-cache equivalence, hard-exit edge cases, memory
isolation and adaptation, retrieval, mixed-difficulty composition, precision
resolution, compile fallback, scaling presets, training/early stopping,
benchmark reports, the SDK, and the `/api` and `/v1` routes.

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

Training also exposes `precision` (`fp32`, `bf16`, `fp16`, `auto`),
`torch_compile`/`compile_mode`, and `mixed_easy_ratio` for exact mixed-difficulty
composition. `adaptive_inference_steps` caps adaptive modes at inference without
changing the training depth; the reference checkpoint uses 3. Parameter-scaled
presets are defined in `SCALING_PRESETS` and can be selected with
`--scaling-preset 15m|50m|125m`. Presets only set architecture fields; the
benchmark still instantiates the model and reports the exact count.

Environment variables read by the backend:

| Variable | Purpose |
|---|---|
| `WILL_DEVICE` | `auto`, `cpu` or `cuda` |
| `WILL_HOME` | base directory for writable state when installed outside a source checkout |
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
| Benchmarks | `GET /api/benchmarks/suites`, `/jobs`, `/jobs/{id}`, `/jobs/{id}/events`, `/reports`, `/reports/{run_id}`, `/reports/{run_id}/json`, `/reports/{run_id}/csv`; `POST /start`, `/jobs/{id}/cancel` |
| Memory | session CRUD under `/api/memory/sessions`; ingest, ingest-text, documents, query, search, adapt, save, restore, reset |
| Datasets | `GET /api/datasets`, `/api/datasets/{task}/sample` |
| Playground | `GET /api/playground/presets`; `POST /tokenize`, `/generate`, `/forward` |
| Experiments | `GET /api/experiments`, `/api/experiments/{name}` |
| OpenAI-compatible | `GET /v1/models`, `/v1/telemetry`; `POST /v1/chat/completions` (streaming SSE), `/v1/memory/absorb`, `/v1/memory/query` |

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
| `mixed_difficulty` | exact 70/30 trivial/hard composition with per-stratum accuracy and loss |
| `memory_recall` | lexical/neural/hybrid recall before and after adaptation |
| `continual` | sequential 3×20-fact learning, replay consolidation, retention and save/load |
| `inference` | latency, throughput, token evaluations, estimated MACs and a `torch.compile` probe |
| `scaling` | 15M/50M/125M presets with exact counts and real fp32/bf16/fp16 steps |
| `ablation` | execution-mode and memory-subsystem comparison |

All eight suites are implemented; the Phase 1–3 reports cited above have been
run against the reported checkpoints. The run IDs are listed under
[Reproducibility](#reproducibility). The `scaling` suite is deliberately
step-level: it validates parameter scaling and precision execution, not
convergence at 125M on CPU.

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
will_engine/    installable SDK: WILLForCausalLM, chat formatting, telemetry
backend/
  engine/       tokenizer, embeddings, context encoder, recurrent block,
                halting gate, controller, output head, unified model, KV cache,
                compilation wrapper, metrics
  memory/       fast weights, memory encoder/manager, BM25, ingestion, retrieval
  training/     datasets, losses, trainer, precision plans, checkpoints
  benchmarks/   suite runners and JSON/CSV report writing
  server/       FastAPI app, /api and /v1 routes, SQLite store, jobs, schemas
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
pyproject.toml  installable package metadata and will-serve entry point
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

- Windows, Python 3.14.4;
- PyTorch 2.14.0+cpu, CUDA unavailable (no CUDA device or CUDA build);
- 4 logical CPUs, 2 PyTorch threads;
- seed 1337 for training datasets;
- tiny profile for the adaptive/memory/mixed reports: 581,765 parameters;
- high process priority and CPU affinity `0xE` for the formal inference run;
- no power meter.

Key report IDs:

| Report | Run ID |
|---|---|
| Cached adaptive compute | `adaptive_compute-1795066048` |
| Continual learning (3×20 facts) | `continual-1795067513` |
| Mixed 70/30 difficulty | `mixed_difficulty-1795166057` |
| Parameter scaling 15M/50M/125M | `scaling-1795164427` |
| Memory recall (hybrid fix) | `memory_recall-1795238098` |
| Inference latency + compile probe | `inference-1795166202` |

The JSON reports are the source of truth for the tables above. Regenerate them
rather than copying numbers between machines. The inference and adaptive-compute
numbers are sensitive to background load and CPU frequency; the reports record
the options and repeats, but not the process priority/affinity, so compare
within a report rather than across unrelated runs.

---

## v4 architecture and WILL-8M (negative result)

The v4 pass targeted three failure modes from `synergy_v3_eval-90552d60aa6f`:
unconditional ponder pressure, a scalar memory gate, and a capacity-starved
581K model. The implementation is in `backend/engine/model.py`,
`backend/training/losses.py`, `backend/memory/memory_encoder.py`, and
`configs/will_8m.yaml`; the curriculum is `scripts/train_will8m.py`. The
recorded evidence run used 1,600 mixed examples (800 hard) and a reduced CPU
budget; the config is a template, not a claim that the full budget was run.

What changed:

- **Accuracy-gated ponder with a depth target.** The early-exit reward applies
  only to fully-correct sequences, and incorrect sequences target `max_steps`
  through a per-token depth loss. A pure `exp(-CE)` gate was measured and
  rejected: hard examples are often *confidently wrong* (median CE `0.004`), so
  confidence alone cannot detect them.
- **Writer-aligned memory query.** The query is pooled from the same
  `memory_encoder` key path used by the Test 5/6 writer, then refined by a
  two-layer `nn.Linear -> SiLU -> nn.Linear` bridge and gated per channel. The
  residual is layer-normalised with a learned scale. The Test 6 writer is the
  exact closed-form ridge consolidation used by Test 5
  (`writer: test5_ridge_consolidate`).
- **WILL-8M.** `configs/will_8m.yaml` builds 8,044,903 slow parameters
  (hidden 400, 2 context layers, `memory_dim=128`) with a 66,048-byte fast state.

Measured result, `will8m_eval-cbb15aeef50f` (5 warmups, 50 timed runs,
CPU, same process):

| Checkpoint | Easy acc | Easy depth | Hard acc | Hard depth | Easy p50 | Hard p50 |
|---|---:|---:|---:|---:|---:|---:|
| old `add-mixed-70-30-v2-final` | 100.0% | 2.107 | 11.0% | 2.139 | 112.47 ms | 158.66 ms |
| new `will-8m-synergy-v1` | 80.0% | 1.268 | 13.0% | 1.891 | 785.30 ms | 1103.51 ms |

| Test 6 | Cold acc | Warm acc | Cold depth | Warm depth | Cold p50 | Warm p50 | Δ depth | Δ p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old | 13.33% | 13.33% | 2.138 | 2.128 | 79.86 ms | 104.99 ms | -0.010 | +25.13 ms |
| new | 10.00% | 16.67% | 1.920 | 1.430 | 365.52 ms | 456.80 ms | -0.490 | +91.28 ms |

The targets were **not** met: easy accuracy `80% < 98%`, hard accuracy
`13% < 85%`, hard depth `1.89 < 2.8`, and Test 6 warm accuracy
`16.67% < 90%`. The new model also costs roughly 7x the old p50 latency
because it is 13x larger and runs four adaptive steps. The memory path is
mechanically correct (query-key and retrieved-value cosine both `1.0` under the
ridge writer), and warm memory now improves Test 6 accuracy (`10% -> 16.67%`)
and reduces depth by `0.49`; the remaining bottleneck is decoding a pooled
value vector into a multi-digit answer, not retrieval. The earlier v3 evidence
remains at `synergy_v3_eval-90552d60aa6f`.

---

## Limitations

- **Synthetic tasks only.** The checkpoints are trained on generated addition,
  multi-hop, sequence and long-context tasks. They are not general-purpose
  language models.
- **Small evaluation sets.** The adaptive-compute split has 150 examples; the
  mixed-difficulty hard stratum has 150 examples; the inference report uses 50
  repeats but only one generated token per prompt. Treat the numbers as
  directional, not statistically strong.
- **Hard ACT speedup is conditional.** With incremental KV caching it is 2.08×
  faster than fixed depth 4 in the adaptive-compute report and faster at all
  three prompt lengths in the controlled inference run. Under background load
  the same comparison produced much smaller and sometimes reversed differences.
  It is not a universal CPU or GPU speedup guarantee.
- **Mixed hard accuracy is low.** The mixed checkpoint solves 100% of the
  trivial stratum but only 10.7–12.7% of the two-digit hard stratum. The 73%
  overall number is dominated by the 70% trivial share; the per-stratum table is
  the meaningful result.
- **No CUDA measurement.** PyTorch is CPU-only here. `bf16`/`fp16` autocast runs
  are real, but on this CPU they are slower than fp32 because bf16 has no native
  path. CUDA support is implemented but unvalidated on this machine.
- **`torch.compile` fell back.** The probe failed with a missing C++ compiler
  (`cl`), so all reported generation timings are eager. No compiled latency is
  claimed.
- **125M is step-validated, not trained.** The scaling suite runs real
  forward/backward steps at 125M; it does not show convergence or downstream
  quality at that size.
- **No energy claim.** There is no power meter in the loop. Estimated MACs are
  not energy.
- **Adaptive soft does not skip work.** It is the differentiable training path;
  only `adaptive_hard` performs hard token skipping.
- **Memory recall is task-specific.** All three retrieval paths reach 100% on a
  20-record synthetic key-value set after 20 rounds. This is not evidence of
  general retrieval quality; lexical BM25 is a strong baseline there.
- **Old checkpoints are not comparable.** Early checkpoints predate the
  answer-token/EOS-only loss masking and the cached generation path. Only
  checkpoints trained with the current loss should be compared.
- **PDF ingestion is text extraction, not OCR.** Scanned image-only PDFs may
  yield no text.
- **Single shared API key.** `WILL_API_KEY` is authentication for a trusted local
  deployment, not a multi-user authorization system.
- **WILL-8M did not meet its targets.** Easy accuracy is 80%, hard accuracy is
  13%, and Test 6 warm accuracy is 16.67% in
  `will8m_eval-cbb15aeef50f`. The depth and accuracy goals are research
  results, not benchmark artifacts; do not cite the model as solving two-digit
  addition.
- **Memory retrieval is exact; memory decoding is not.** The ridge writer
  retrieves the stored value with cosine `1.0`, but the pooled value vector is
  not reliably decoded into a multi-digit answer. This is the main open
  bottleneck for Test 6.
- **Size and latency are not comparable across checkpoints.** WILL-8M is 13x
  larger than the 581K reference and is roughly 7x slower at p50 on this CPU.
  The table reports the measurement; it is not a speed claim.

---

## License

MIT. See `LICENSE`.
