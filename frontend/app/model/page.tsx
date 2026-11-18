"use client";

import { useState } from "react";

import { BarRows } from "@/components/Charts";
import {
  Badge,
  Btn,
  Empty,
  ErrorBox,
  KV,
  Loading,
  PageHeader,
  Panel,
  Stat,
} from "@/components/ui";
import { fmtCompact, fmtNum, fmtPct, post } from "@/lib/api";
import { useApi } from "@/lib/hooks";
import type { CheckpointMeta, ModelSummary } from "@/lib/types";

export default function ModelPage() {
  const current = useApi<ModelSummary>("/api/checkpoints/current", []);
  const checkpoints = useApi<{
    checkpoints: CheckpointMeta[];
    default_checkpoint_id: string | null;
  }>("/api/checkpoints", []);
  const config = useApi<Record<string, any>>("/api/config", []);
  const [selecting, setSelecting] = useState(false);
  const [selectError, setSelectError] = useState<string | null>(null);

  const summary = current.data;
  const cfg = summary?.model_config;
  const training = summary?.training_config || {};
  const breakdown = summary?.module_breakdown || [];

  async function selectCheckpoint(id: string) {
    if (!id) return;
    setSelecting(true);
    setSelectError(null);
    try {
      await post(`/api/checkpoints/${id}/select`, {});
      current.reload();
      checkpoints.reload();
    } catch (err) {
      setSelectError((err as Error).message);
    } finally {
      setSelecting(false);
    }
  }

  return (
    <>
      <PageHeader
        kicker="WILL / MODEL"
        title="One shared block, variable depth."
        description="The recurrent reasoner reuses a single transformer block at every step. A halting controller allocates steps per token; a separate fast-weight memory can be attached to a session without touching slow parameters."
        actions={
          <div className="field" style={{ minWidth: 280 }}>
            <label>Active checkpoint</label>
            <select
              value={summary?.checkpoint_id || ""}
              onChange={(event) => selectCheckpoint(event.target.value)}
              disabled={selecting || !checkpoints.data?.checkpoints?.length}
            >
              {!checkpoints.data?.checkpoints?.length ? (
                <option value="">No checkpoints</option>
              ) : null}
              {checkpoints.data?.checkpoints?.map((ckpt) => (
                <option value={ckpt.checkpoint_id} key={ckpt.checkpoint_id}>
                  {ckpt.checkpoint_id}
                </option>
              ))}
            </select>
          </div>
        }
      />

      {selectError ? <ErrorBox error={selectError} /> : null}
      {current.loading && !summary ? <Loading label="Loading model summary" /> : null}
      {current.error ? <ErrorBox error={current.error} onRetry={current.reload} /> : null}

      {summary && cfg ? (
        <>
          <div className="stat-grid">
            <Stat
              label="Total parameters"
              value={fmtNum(summary.parameters_total)}
              small="slow weights"
              foot={`${fmtNum(summary.parameters_trainable)} trainable`}
              tone="mint"
            />
            <Stat
              label="Frozen parameters"
              value={fmtNum(summary.parameters_frozen)}
              small="adaptation never writes"
              foot="fast weights live in session state"
            />
            <Stat
              label="Fast memory init"
              value={fmtNum(summary.memory_initial_parameters)}
              small="state values"
              foot={`${cfg.memory_kind} · dim ${cfg.memory_dim}`}
              tone="cyan"
            />
            <Stat
              label="Recurrent budget"
              value={cfg.max_steps}
              small="steps max"
              foot={`min ${cfg.min_steps} · halt ε ${cfg.halt_epsilon}`}
              tone="amber"
            />
          </div>

          <div className="split" style={{ marginTop: 16 }}>
            <Panel
              title="Architecture"
              subtitle="from the checkpoint's serialized model config"
              actions={<Badge tone="mint">{summary.task || "unknown task"}</Badge>}
            >
              <div className="arch">
                <div className="arch-layer">
                  <span className="name">Input embedding</span>
                  <span className="desc">
                    byte vocabulary {cfg.vocab_size} · hidden {cfg.hidden_dim} · max
                    context {cfg.max_seq_len}
                  </span>
                </div>
                <div className="arch-arrow" />
                <div className="arch-layer">
                  <span className="name">Context encoder</span>
                  <span className="desc">
                    {cfg.context_layers} causal layer{cfg.context_layers === 1 ? "" : "s"} ·{" "}
                    {cfg.n_heads} heads · ff ×{cfg.context_ff_mult}
                  </span>
                </div>
                <div className="arch-arrow" />
                <div className="arch-layer">
                  <span className="name">Recurrent block</span>
                  <span className="desc">
                    shared every step · hidden {cfg.hidden_dim} · ff ×{cfg.ff_mult} ·
                    starts from contextual state
                  </span>
                </div>
                <div className="arch-arrow" />
                <div className="arch-layer">
                  <span className="name">Halting controller</span>
                  <span className="desc">
                    ACT soft training · hard inference gather/scatter · max {cfg.max_steps}{" "}
                    steps · min {cfg.min_steps}
                  </span>
                </div>
                <div className="arch-arrow" />
                <div className="arch-layer">
                  <span className="name">Fast memory</span>
                  <span className="desc">
                    {cfg.memory_kind} · {cfg.memory_dim} dims · inner steps{" "}
                    {cfg.memory_inner_steps} · lr {cfg.memory_inner_lr}
                  </span>
                </div>
                <div className="arch-arrow" />
                <div className="arch-layer">
                  <span className="name">Output projection</span>
                  <span className="desc">
                    hidden {cfg.hidden_dim} → byte vocabulary {cfg.vocab_size} ·
                    answer-only loss masking during training
                  </span>
                </div>
              </div>
            </Panel>

            <div className="stack">
              <Panel title="Checkpoint" subtitle="selected slow-weight state">
                <KV
                  rows={[
                    ["checkpoint", <span key="id" className="mono">{summary.checkpoint_id || "fresh model"}</span>],
                    ["task", summary.task || "—"],
                    ["device", summary.device],
                    ["best val loss", training.best_val_loss ? fmtNum(training.best_val_loss, 4) : summary.metrics?.best_val_loss ? fmtNum(summary.metrics.best_val_loss, 4) : "—"],
                    ["best val acc", summary.metrics?.best_val_accuracy ? fmtPct(summary.metrics.best_val_accuracy) : "—"],
                    ["epochs", String(training.epochs ?? "—")],
                    ["mode", String(training.mode ?? "—")],
                    ["seed", String(training.seed ?? "—")],
                  ]}
                />
              </Panel>
              <Panel title="Parameter mass" subtitle="direct module parameters">
                {breakdown.length ? (
                  <BarRows
                    rows={breakdown.map((row) => ({
                      label: row.name,
                      value: row.parameters,
                    }))}
                    format={(value) => fmtCompact(value)}
                  />
                ) : (
                  <Empty>No module breakdown in this checkpoint.</Empty>
                )}
              </Panel>
            </div>
          </div>

          <div className="section-title">
            <h2>Subsystem contract</h2>
            <span>separately testable</span>
          </div>

          <div className="grid grid-3">
            <Panel title="Adaptive computation" subtitle="controller + shared block">
              <p className="panel-sub" style={{ lineHeight: 1.7 }}>
                <strong style={{ color: "var(--text)" }}>adaptive_soft</strong> trains
                every step with a differentiable ponder cost.{" "}
                <strong style={{ color: "var(--text)" }}>adaptive_hard</strong> runs
                under <span className="mono">no_grad</span>, gathers only active tokens
                and skips halted positions. Fixed depth remains a baseline.
              </p>
              <div className="row" style={{ marginTop: 12 }}>
                <Badge tone="mint">fixed_depth</Badge>
                <Badge tone="cyan">adaptive_soft</Badge>
                <Badge tone="amber">adaptive_hard</Badge>
              </div>
            </Panel>
            <Panel title="Fast-weight memory" subtitle="session state, not slow weights">
              <p className="panel-sub" style={{ lineHeight: 1.7 }}>
                TTT and associative memory adapt into a bounded{" "}
                <span className="mono">FastMemoryState</span>. Sessions are UUID-isolated,
                serializable and restorable. Slow parameters are frozen during
                adaptation by construction.
              </p>
              <div className="row" style={{ marginTop: 12 }}>
                <Badge tone="mint">bounded</Badge>
                <Badge tone="cyan">serializable</Badge>
                <Badge tone="amber">session-isolated</Badge>
              </div>
            </Panel>
            <Panel title="Memory-off path" subtitle="unified model remains functional">
              <p className="panel-sub" style={{ lineHeight: 1.7 }}>
                Every training and evaluation path can disable memory integration. The
                unified model then reduces to the adaptive recurrent reasoner, which is
                the configuration used for the arithmetic depth sweep.
              </p>
              <div className="row" style={{ marginTop: 12 }}>
                <Badge tone="mint">memory_enabled=false</Badge>
                <Badge>tested</Badge>
              </div>
            </Panel>
          </div>

          <div className="section-title">
            <h2>Serialized configuration</h2>
            <span>read-only</span>
          </div>
          <div className="grid grid-2">
            <Panel title="ModelConfig" flush>
              <pre style={{ border: "none", borderRadius: 0, maxHeight: 420 }}>
                {JSON.stringify(cfg, null, 2)}
              </pre>
            </Panel>
            <Panel title="TrainingConfig" flush>
              <pre style={{ border: "none", borderRadius: 0, maxHeight: 420 }}>
                {JSON.stringify(training, null, 2)}
              </pre>
            </Panel>
          </div>

          {config.data ? (
            <div style={{ marginTop: 16 }}>
              <Panel title="Server profile" subtitle="paths and runtime defaults" tight>
                <KV
                  rows={[
                    ["profile", String(config.data.profile ?? "—")],
                    ["device", String(config.data.server?.device ?? "—")],
                    ["checkpoint dir", <span key="p" className="mono">{String(config.data.server?.checkpoint_dir ?? "—")}</span>],
                    ["experiment dir", <span key="e" className="mono">{String(config.data.server?.experiment_dir ?? "—")}</span>],
                    ["max upload", `${fmtNum(config.data.memory?.max_upload_bytes)} bytes`],
                  ]}
                />
              </Panel>
            </div>
          ) : null}
        </>
      ) : null}
    </>
  );
}
