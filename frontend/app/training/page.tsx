"use client";

import { useMemo, useState } from "react";

import { LineChart } from "@/components/Charts";
import {
  Badge,
  Btn,
  Empty,
  ErrorBox,
  Field,
  Loading,
  PageHeader,
  Panel,
  ProgressBar,
  StatusBadge,
} from "@/components/ui";
import {
  fmtAgo,
  fmtNum,
  fmtPct,
  post,
} from "@/lib/api";
import { useApi, useJob } from "@/lib/hooks";
import type { CheckpointMeta, Job, JobEvent } from "@/lib/types";

type Defaults = {
  training: Record<string, any>;
  model: Record<string, any>;
  profiles: string[];
  tasks: string[];
};

type JobList = { jobs: Job[] };

type FormState = {
  task: string;
  profile: string;
  mode: "fixed_depth" | "adaptive_soft";
  epochs: number;
  train_size: number;
  val_size: number;
  test_size: number;
  batch_size: number;
  lr: number;
  warmup_steps: number;
  max_steps: number;
  min_steps: number;
  halt_epsilon: number;
  ponder_lambda: number;
  ponder_warmup_epochs: number;
  min_complexity: number;
  max_complexity: number;
  seed: number;
  checkpoint_name: string;
};

const INITIAL: FormState = {
  task: "arith_add",
  profile: "tiny",
  mode: "adaptive_soft",
  epochs: 30,
  train_size: 1000,
  val_size: 150,
  test_size: 150,
  batch_size: 16,
  lr: 0.0005,
  warmup_steps: 20,
  max_steps: 4,
  min_steps: 2,
  halt_epsilon: 0.05,
  ponder_lambda: 0.01,
  ponder_warmup_epochs: 10,
  min_complexity: 1,
  max_complexity: 1,
  seed: 1337,
  checkpoint_name: "",
};

function eventLine(event: JobEvent): string {
  if (event.type === "step") {
    return `epoch ${event.epoch ?? "—"} step ${event.step ?? "—"} loss ${event.loss ?? "—"} depth ${event.mean_depth ?? "—"}`;
  }
  if (event.type === "eval") {
    return `eval epoch ${event.epoch ?? "—"} val_loss ${event.val_loss ?? "—"} val_acc ${event.val_accuracy ?? "—"} depth ${event.val_mean_depth ?? "—"}`;
  }
  if (event.type === "checkpoint") {
    return `checkpoint ${event.checkpoint_id}`;
  }
  if (event.type === "start_training") {
    return `start task=${event.task} mode=${event.mode} max_steps=${event.max_steps} params=${event.parameters}`;
  }
  if (event.type === "early_stop") {
    return `early stop at epoch ${event.epoch}`;
  }
  if (event.type === "error") {
    return `error: ${event.message}`;
  }
  return `${event.type}: ${event.message ?? event.label ?? JSON.stringify(event).slice(0, 220)}`;
}

export default function TrainingPage() {
  const defaults = useApi<Defaults>("/api/training/defaults", []);
  const jobs = useApi<JobList>("/api/training/jobs", []);
  const checkpoints = useApi<{ checkpoints: CheckpointMeta[] }>("/api/checkpoints", []);
  const [form, setForm] = useState<FormState>(INITIAL);
  const [jobId, setJobId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState<string | null>(null);
  const { job, events, error: pollError } = useJob("training", jobId);

  const history = useMemo(() => {
    const resultHistory = job?.result?.history;
    if (Array.isArray(resultHistory) && resultHistory.length) return resultHistory;
    return events.filter((event) => event.type === "eval");
  }, [job?.result?.history, events]);

  const chartX = history.map((entry: any) => `e${entry.epoch ?? "?"}`);

  function set<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  async function start() {
    setStarting(true);
    setError(null);
    try {
      const payload = {
        ...form,
        checkpoint_name: form.checkpoint_name.trim() || undefined,
      };
      const response = await post<{ job_id: string }>("/api/training/start", payload);
      setJobId(response.job_id);
      jobs.reload();
      checkpoints.reload();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setStarting(false);
    }
  }

  async function cancel(id: string) {
    setCancelling(id);
    setError(null);
    try {
      await post(`/api/training/jobs/${id}/cancel`, {});
      jobs.reload();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setCancelling(null);
    }
  }

  return (
    <>
      <PageHeader
        kicker="WILL / TRAINING"
        title="Train the gate, not just the weights."
        description="Launch a real PyTorch training job. The adaptive controller is trained with a differentiable ponder cost; hard early exit is reserved for inference so the training graph remains differentiable."
        actions={
          <>
            <Badge tone="cyan">CPU / CUDA</Badge>
            <Btn onClick={() => jobs.reload()}>Refresh jobs</Btn>
          </>
        }
      />

      {error ? <ErrorBox error={error} /> : null}

      <div className="split">
        <Panel
          title="New run"
          subtitle="configuration is sent to the FastAPI job worker"
          actions={
            <Btn variant="primary" onClick={start} disabled={starting}>
              {starting ? "Queueing…" : "Start training"}
            </Btn>
          }
        >
          <div className="form-grid">
            <Field label="Task">
              <select value={form.task} onChange={(e) => set("task", e.target.value)}>
                {(defaults.data?.tasks || [
                  "arith_add",
                  "arithmetic",
                  "sequence",
                  "multihop",
                  "longctx",
                  "lm",
                  "mixture",
                ]).map((task) => (
                  <option key={task} value={task}>
                    {task}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Profile">
              <select value={form.profile} onChange={(e) => set("profile", e.target.value)}>
                {(defaults.data?.profiles || ["tiny", "standard", "experimental", "custom"]).map(
                  (profile) => (
                    <option key={profile} value={profile}>
                      {profile}
                    </option>
                  ),
                )}
              </select>
            </Field>
            <Field label="Execution mode">
              <select
                value={form.mode}
                onChange={(e) => set("mode", e.target.value as FormState["mode"])}
              >
                <option value="adaptive_soft">adaptive_soft (differentiable ACT)</option>
                <option value="fixed_depth">fixed_depth (baseline)</option>
              </select>
            </Field>
            <Field label="Epochs">
              <input
                type="number"
                min={1}
                max={500}
                value={form.epochs}
                onChange={(e) => set("epochs", Number(e.target.value))}
              />
            </Field>
            <Field label="Train examples">
              <input
                type="number"
                min={1}
                value={form.train_size}
                onChange={(e) => set("train_size", Number(e.target.value))}
              />
            </Field>
            <Field label="Val / test examples">
              <div className="row" style={{ flexWrap: "nowrap" }}>
                <input
                  type="number"
                  min={1}
                  value={form.val_size}
                  onChange={(e) => set("val_size", Number(e.target.value))}
                />
                <input
                  type="number"
                  min={1}
                  value={form.test_size}
                  onChange={(e) => set("test_size", Number(e.target.value))}
                />
              </div>
            </Field>
            <Field label="Batch size">
              <input
                type="number"
                min={1}
                value={form.batch_size}
                onChange={(e) => set("batch_size", Number(e.target.value))}
              />
            </Field>
            <Field label="Learning rate">
              <input
                type="number"
                step="0.0001"
                min={0.000001}
                value={form.lr}
                onChange={(e) => set("lr", Number(e.target.value))}
              />
            </Field>
            <Field label="Warmup steps">
              <input
                type="number"
                min={0}
                value={form.warmup_steps}
                onChange={(e) => set("warmup_steps", Number(e.target.value))}
              />
            </Field>
            <Field label="Max steps">
              <input
                type="number"
                min={1}
                max={64}
                value={form.max_steps}
                onChange={(e) => set("max_steps", Number(e.target.value))}
              />
            </Field>
            <Field label="Min steps">
              <input
                type="number"
                min={1}
                max={64}
                value={form.min_steps}
                onChange={(e) => set("min_steps", Number(e.target.value))}
              />
            </Field>
            <Field label="Halt ε">
              <input
                type="number"
                step="0.01"
                min={0.001}
                max={0.999}
                value={form.halt_epsilon}
                onChange={(e) => set("halt_epsilon", Number(e.target.value))}
              />
            </Field>
            <Field label="Ponder λ">
              <input
                type="number"
                step="0.001"
                min={0}
                value={form.ponder_lambda}
                onChange={(e) => set("ponder_lambda", Number(e.target.value))}
              />
            </Field>
            <Field label="Ponder warmup epochs">
              <input
                type="number"
                min={0}
                value={form.ponder_warmup_epochs}
                onChange={(e) => set("ponder_warmup_epochs", Number(e.target.value))}
              />
            </Field>
            <Field label="Complexity min / max">
              <div className="row" style={{ flexWrap: "nowrap" }}>
                <input
                  type="number"
                  min={1}
                  max={4}
                  value={form.min_complexity}
                  onChange={(e) => set("min_complexity", Number(e.target.value))}
                />
                <input
                  type="number"
                  min={1}
                  max={4}
                  value={form.max_complexity}
                  onChange={(e) => set("max_complexity", Number(e.target.value))}
                />
              </div>
            </Field>
            <Field label="Seed">
              <input
                type="number"
                min={0}
                value={form.seed}
                onChange={(e) => set("seed", Number(e.target.value))}
              />
            </Field>
            <Field label="Checkpoint name" hint="blank = timestamped run name">
              <input
                value={form.checkpoint_name}
                onChange={(e) => set("checkpoint_name", e.target.value)}
                placeholder="e.g. add-single-act"
              />
            </Field>
          </div>
          <div className="notice" style={{ marginTop: 14 }}>
            The successful ACT recipe used min steps 2, ponder λ 0.01 and 10 warmup
            epochs. Lower λ saves less compute; a zero warmup can collapse the gate to
            one step before the task is learned.
          </div>
        </Panel>

        <div className="stack">
          <Panel
            title="Active job"
            subtitle={jobId ? jobId : "no job selected"}
            actions={job ? <StatusBadge status={job.status} /> : null}
          >
            {!jobId ? (
              <Empty>Start a run or open a recent job.</Empty>
            ) : pollError ? (
              <ErrorBox error={pollError} />
            ) : !job ? (
              <Loading label="Attaching to job" />
            ) : (
              <>
                <div className="spread" style={{ marginBottom: 8 }}>
                  <span className="mono" style={{ fontSize: 11 }}>
                    {job.message || job.status}
                  </span>
                  <span className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>
                    {Math.round((job.progress || 0) * 100)}%
                  </span>
                </div>
                <ProgressBar value={job.progress || 0} />
                <div className="row" style={{ marginTop: 12 }}>
                  <Badge tone="cyan">{job.config?.mode || "—"}</Badge>
                  <Badge>{job.config?.task || "—"}</Badge>
                  <Badge>{job.config?.epochs ?? "—"} epochs</Badge>
                  <Badge>{job.config?.max_steps ?? "—"} max steps</Badge>
                  {job.status === "running" || job.status === "queued" ? (
                    <Btn
                      size="sm"
                      variant="danger"
                      onClick={() => cancel(job.id)}
                      disabled={cancelling === job.id}
                    >
                      Cancel
                    </Btn>
                  ) : null}
                </div>
                {job.error ? (
                  <div className="error-box" style={{ marginTop: 12, whiteSpace: "pre-wrap" }}>
                    {job.error}
                  </div>
                ) : null}
                {job.result ? (
                  <div className="kv" style={{ marginTop: 14 }}>
                    <div className="k">checkpoint</div>
                    <div className="v">{job.result.checkpoint_id || "—"}</div>
                    <div className="k">best val loss</div>
                    <div className="v">{fmtNum(job.result.best_val_loss, 4)}</div>
                    <div className="k">best val accuracy</div>
                    <div className="v">
                      {job.result.best_val_accuracy === null
                        ? "—"
                        : fmtPct(job.result.best_val_accuracy)}
                    </div>
                    <div className="k">duration</div>
                    <div className="v">{fmtNum(job.result.duration_seconds, 1)} s</div>
                  </div>
                ) : null}
              </>
            )}
          </Panel>

          <Panel title="Live telemetry" subtitle="emitted by the training loop">
            {history.length ? (
              <>
                <LineChart
                  series={[
                    {
                      label: "train loss",
                      values: history.map((entry: any) => entry.train_loss ?? null),
                      color: "#64d8ff",
                    },
                    {
                      label: "val loss",
                      values: history.map((entry: any) => entry.val_loss ?? null),
                      color: "#7cf5c8",
                    },
                  ]}
                  xLabels={chartX}
                  height={170}
                />
                <div style={{ marginTop: 12 }}>
                  <LineChart
                    series={[
                      {
                        label: "val accuracy",
                        values: history.map((entry: any) => entry.val_accuracy ?? null),
                        color: "#f6c177",
                      },
                    ]}
                    xLabels={chartX}
                    height={145}
                    yFormat={(value) => `${(value * 100).toFixed(0)}%`}
                    zeroBase
                  />
                </div>
              </>
            ) : (
              <Empty>Waiting for the first evaluation point.</Empty>
            )}
          </Panel>
        </div>
      </div>

      <div className="section-title">
        <h2>Event log</h2>
        <span>{events.length} persisted events</span>
      </div>
      <Panel flush>
        <div className="log">
          {events.length ? (
            events.slice(-120).map((event, index) => (
              <div
                className={`log-line ${event.type === "error" ? "err" : event.type === "done" ? "ok" : ""}`}
                key={`${event._event_id ?? index}-${event.type}`}
              >
                <span className="t">{event.type}</span>
                <span>{eventLine(event)}</span>
              </div>
            ))
          ) : (
            <div className="empty">No events yet.</div>
          )}
        </div>
      </Panel>

      <div className="section-title">
        <h2>Recent training jobs</h2>
        <span>SQLite-backed</span>
      </div>
      <Panel flush>
        {jobs.loading && !jobs.data ? (
          <Loading />
        ) : jobs.error ? (
          <ErrorBox error={jobs.error} onRetry={jobs.reload} />
        ) : jobs.data?.jobs?.length ? (
          <div className="table-wrap" style={{ border: "none" }}>
            <table>
              <thead>
                <tr>
                  <th>job</th>
                  <th>status</th>
                  <th>task</th>
                  <th>mode</th>
                  <th className="num">epochs</th>
                  <th className="num">best acc</th>
                  <th>checkpoint</th>
                  <th>created</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {jobs.data.jobs.map((row) => (
                  <tr key={row.id}>
                    <td className="mono">{row.id.slice(0, 10)}</td>
                    <td><StatusBadge status={row.status} /></td>
                    <td>{row.config?.task || "—"}</td>
                    <td className="mono">{row.config?.mode || "—"}</td>
                    <td className="num">{row.config?.epochs ?? "—"}</td>
                    <td className="num">
                      {row.result?.best_val_accuracy === undefined ||
                      row.result?.best_val_accuracy === null
                        ? "—"
                        : fmtPct(row.result.best_val_accuracy)}
                    </td>
                    <td className="mono">{row.result?.checkpoint_id || "—"}</td>
                    <td className="mono">{fmtAgo(row.created_at)}</td>
                    <td>
                      <div className="row" style={{ flexWrap: "nowrap" }}>
                        <Btn size="sm" onClick={() => setJobId(row.id)}>
                          Open
                        </Btn>
                        {row.status === "running" || row.status === "queued" ? (
                          <Btn
                            size="sm"
                            variant="danger"
                            onClick={() => cancel(row.id)}
                            disabled={cancelling === row.id}
                          >
                            Cancel
                          </Btn>
                        ) : null}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>No training jobs yet.</Empty>
        )}
      </Panel>

      {defaults.error ? (
        <div style={{ marginTop: 16 }}>
          <ErrorBox error={defaults.error} onRetry={defaults.reload} />
        </div>
      ) : null}
    </>
  );
}
