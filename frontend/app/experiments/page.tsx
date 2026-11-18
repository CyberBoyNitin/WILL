"use client";

import { useEffect, useMemo, useState } from "react";

import { LineChart } from "@/components/Charts";
import {
  Badge,
  Btn,
  Empty,
  ErrorBox,
  Field,
  KV,
  Loading,
  PageHeader,
  Panel,
  Stat,
} from "@/components/ui";
import {
  api,
  fmtAgo,
  fmtBytes,
  fmtNum,
  fmtPct,
  post,
} from "@/lib/api";
import { useApi } from "@/lib/hooks";
import type { CheckpointMeta } from "@/lib/types";

type ExperimentItem = {
  name: string;
  size_bytes: number;
  modified_at: number;
  kind: string;
  summary?: Record<string, any> | null;
};

type EvalResult = {
  checkpoint_id: string;
  split: string;
  mode: string;
  max_steps: number;
  loss: Record<string, any>;
  generation: Record<string, any>;
  dataset: Record<string, any>;
};

function experimentPath(name: string) {
  return name
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");
}

export default function ExperimentsPage() {
  const checkpoints = useApi<{
    checkpoints: CheckpointMeta[];
    default_checkpoint_id: string | null;
  }>("/api/checkpoints", []);
  const experiments = useApi<{ experiments: ExperimentItem[] }>(
    "/api/experiments",
    [],
  );
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string | null>(null);
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);
  const [split, setSplit] = useState("test");
  const [limit, setLimit] = useState(100);
  const [evalMode, setEvalMode] = useState("fixed_depth");
  const [evalSteps, setEvalSteps] = useState("");
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const meta = useApi<CheckpointMeta>(
    selectedCheckpoint
      ? `/api/checkpoints/${encodeURIComponent(selectedCheckpoint)}`
      : null,
    [selectedCheckpoint],
  );
  const history = useApi<any>(
    selectedCheckpoint
      ? `/api/checkpoints/${encodeURIComponent(selectedCheckpoint)}/history`
      : null,
    [selectedCheckpoint],
  );
  const experimentDetail = useApi<any>(
    selectedExperiment
      ? `/api/experiments/${experimentPath(selectedExperiment)}`
      : null,
    [selectedExperiment],
  );

  useEffect(() => {
    if (!selectedCheckpoint && checkpoints.data?.checkpoints?.length) {
      setSelectedCheckpoint(checkpoints.data.checkpoints[0].checkpoint_id);
    }
  }, [checkpoints.data, selectedCheckpoint]);

  useEffect(() => {
    if (!selectedExperiment && experiments.data?.experiments?.length) {
      setSelectedExperiment(experiments.data.experiments[0].name);
    }
  }, [experiments.data, selectedExperiment]);

  const historyRows = history.data?.history || [];
  const chartX = historyRows.map((row: any) => `e${row.epoch ?? "?"}`);

  const totals = useMemo(() => {
    const items = checkpoints.data?.checkpoints || [];
    const latest = items[0];
    return {
      count: items.length,
      defaultId: checkpoints.data?.default_checkpoint_id || null,
      latestId: latest?.checkpoint_id || null,
      reports: (experiments.data?.experiments || []).filter(
        (item) => item.kind === "benchmark_report",
      ).length,
    };
  }, [checkpoints.data, experiments.data]);

  async function selectCheckpoint(id: string) {
    setBusy(`select-${id}`);
    setError(null);
    try {
      await post(`/api/checkpoints/${encodeURIComponent(id)}/select`, {});
      checkpoints.reload();
      setSelectedCheckpoint(id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function deleteCheckpoint(id: string) {
    if (!window.confirm(`Delete checkpoint ${id} and its metadata?`)) return;
    setBusy(`delete-${id}`);
    setError(null);
    try {
      await api(`/api/checkpoints/${encodeURIComponent(id)}`, {
        method: "DELETE",
      });
      if (selectedCheckpoint === id) {
        setSelectedCheckpoint(null);
        setEvalResult(null);
      }
      checkpoints.reload();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function evaluate() {
    if (!selectedCheckpoint) return;
    setBusy("evaluate");
    setError(null);
    try {
      const params = new URLSearchParams({
        split,
        limit: String(limit),
        mode: evalMode,
      });
      if (evalSteps.trim()) params.set("max_steps", evalSteps);
      const result = await api<EvalResult>(
        `/api/checkpoints/${encodeURIComponent(selectedCheckpoint)}/evaluate?${params.toString()}`,
      );
      setEvalResult(result);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  }

  return (
    <>
      <PageHeader
        kicker="WILL / EXPERIMENTS"
        title="The evidence trail."
        description="Checkpoints, training histories and benchmark artifacts on disk. Evaluate any checkpoint against a real split, select it for the rest of the console, or delete it. Nothing here is a mock result."
        actions={
          <>
            <Badge tone="mint">{totals.count} checkpoints</Badge>
            <Badge tone="cyan">{totals.reports} reports</Badge>
            <Btn onClick={() => checkpoints.reload()}>Refresh</Btn>
          </>
        }
      />

      {error ? <ErrorBox error={error} /> : null}

      <div className="stat-grid">
        <Stat
          label="Checkpoints"
          value={fmtNum(totals.count)}
          small="slow weights"
          foot={totals.latestId || "none"}
          tone="mint"
        />
        <Stat
          label="Default"
          value={totals.defaultId ? totals.defaultId.slice(0, 18) : "—"}
          small={totals.defaultId && totals.defaultId.length > 18 ? "…" : ""}
          foot="used by API generation and memory"
        />
        <Stat
          label="Benchmark reports"
          value={fmtNum(totals.reports)}
          small="JSON + CSV"
          foot="experiment directory"
          tone="cyan"
        />
        <Stat
          label="Experiment files"
          value={fmtNum(experiments.data?.experiments?.length)}
          small="artifacts"
          foot="readable through the API"
          tone="amber"
        />
      </div>

      <div className="section-title">
        <h2>Checkpoint registry</h2>
        <span>select, evaluate, delete</span>
      </div>

      <div className="split">
        <Panel flush>
          {checkpoints.loading && !checkpoints.data ? (
            <Loading />
          ) : checkpoints.error ? (
            <ErrorBox error={checkpoints.error} onRetry={checkpoints.reload} />
          ) : checkpoints.data?.checkpoints?.length ? (
            <div className="table-wrap" style={{ border: "none" }}>
              <table>
                <thead>
                  <tr>
                    <th>checkpoint</th>
                    <th>task</th>
                    <th className="num">best acc</th>
                    <th className="num">best loss</th>
                    <th className="num">params</th>
                    <th>created</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {checkpoints.data.checkpoints.map((ckpt) => {
                    const metrics = ckpt.metrics || {};
                    const acc = metrics.best_val_accuracy ?? metrics.val_accuracy;
                    const loss = metrics.best_val_loss ?? metrics.val_loss;
                    const params = metrics.parameters_total ?? null;
                    return (
                      <tr key={ckpt.checkpoint_id}>
                        <td className="mono">
                          {ckpt.checkpoint_id}
                          {ckpt.checkpoint_id === totals.defaultId ? (
                            <span style={{ marginLeft: 8 }}>
                              <Badge tone="mint">default</Badge>
                            </span>
                          ) : null}
                        </td>
                        <td>{ckpt.task || "—"}</td>
                        <td className="num mono">
                          {typeof acc === "number" ? fmtPct(acc) : "—"}
                        </td>
                        <td className="num mono">
                          {typeof loss === "number" ? fmtNum(loss, 4) : "—"}
                        </td>
                        <td className="num mono">{params ? fmtNum(params) : "—"}</td>
                        <td className="mono">{fmtAgo(ckpt.created_at)}</td>
                        <td>
                          <div className="row" style={{ flexWrap: "nowrap" }}>
                            <Btn
                              size="sm"
                              onClick={() => {
                                setSelectedCheckpoint(ckpt.checkpoint_id);
                                setEvalResult(null);
                              }}
                            >
                              Inspect
                            </Btn>
                            <Btn
                              size="sm"
                              onClick={() => selectCheckpoint(ckpt.checkpoint_id)}
                              disabled={busy === `select-${ckpt.checkpoint_id}`}
                            >
                              Select
                            </Btn>
                            <Btn
                              size="sm"
                              variant="danger"
                              onClick={() => deleteCheckpoint(ckpt.checkpoint_id)}
                              disabled={busy === `delete-${ckpt.checkpoint_id}`}
                            >
                              Delete
                            </Btn>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty>No checkpoints on disk.</Empty>
          )}
        </Panel>

        <Panel
          title="Evaluate checkpoint"
          subtitle={selectedCheckpoint || "select a checkpoint"}
          actions={<Badge tone="cyan">{evalMode}</Badge>}
        >
          <div className="form-grid two">
            <Field label="Split">
              <select value={split} onChange={(event) => setSplit(event.target.value)}>
                <option value="test">test</option>
                <option value="val">val</option>
                <option value="train">train</option>
              </select>
            </Field>
            <Field label="Limit">
              <input
                type="number"
                min={1}
                max={1000}
                value={limit}
                onChange={(event) => setLimit(Number(event.target.value))}
              />
            </Field>
            <Field label="Mode">
              <select
                value={evalMode}
                onChange={(event) => setEvalMode(event.target.value)}
              >
                <option value="fixed_depth">fixed_depth</option>
                <option value="adaptive_soft">adaptive_soft</option>
                <option value="adaptive_hard">adaptive_hard</option>
              </select>
            </Field>
            <Field label="Max steps" hint="blank = checkpoint default">
              <input
                type="number"
                min={1}
                max={64}
                value={evalSteps}
                onChange={(event) => setEvalSteps(event.target.value)}
                placeholder="default"
              />
            </Field>
          </div>
          <div className="row" style={{ marginTop: 12 }}>
            <Btn
              variant="primary"
              onClick={evaluate}
              disabled={!selectedCheckpoint || busy === "evaluate"}
            >
              {busy === "evaluate" ? "Evaluating…" : "Evaluate"}
            </Btn>
            <span className="panel-sub">
              Runs loss and generation on the checkpoint&apos;s training task.
            </span>
          </div>

          {evalResult ? (
            <div style={{ marginTop: 14 }}>
              <div className="grid grid-2">
                <KV
                  rows={[
                    ["loss", fmtNum(evalResult.loss.loss, 4)],
                    ["perplexity", fmtNum(evalResult.loss.perplexity, 3)],
                    ["mean depth", fmtNum(evalResult.loss.mean_depth, 3)],
                    ["token evals", fmtNum(evalResult.loss.token_evaluations)],
                    ["est. MACs", fmtNum(evalResult.loss.estimated_macs)],
                    ["sequences", fmtNum(evalResult.loss.sequences)],
                  ]}
                />
                <KV
                  rows={[
                    ["accuracy", fmtPct(evalResult.generation.accuracy)],
                    ["examples", fmtNum(evalResult.generation.examples)],
                    ["gen depth", fmtNum(evalResult.generation.mean_depth, 3)],
                    ["gen token evals", fmtNum(evalResult.generation.token_evaluations)],
                    ["gen seconds", fmtNum(evalResult.generation.elapsed_seconds, 2)],
                    ["max steps", String(evalResult.max_steps)],
                  ]}
                />
              </div>
              <div className="row" style={{ marginTop: 10 }}>
                <Badge tone="mint">{evalResult.split}</Badge>
                <Badge>{evalResult.mode}</Badge>
                <Badge>{evalResult.dataset.task}</Badge>
                <Badge>{evalResult.dataset.size} examples</Badge>
              </div>
              <details style={{ marginTop: 12 }}>
                <summary
                  className="mono"
                  style={{ color: "var(--muted)", cursor: "pointer", fontSize: 10 }}
                >
                  RAW EVALUATION JSON
                </summary>
                <pre style={{ marginTop: 8, maxHeight: 240 }}>
                  {JSON.stringify(evalResult, null, 2)}
                </pre>
              </details>
            </div>
          ) : (
            <div style={{ marginTop: 14 }}>
              <Empty>No evaluation run yet.</Empty>
            </div>
          )}
        </Panel>
      </div>

      <div className="split" style={{ marginTop: 16 }}>
        <Panel
          title="Training history"
          subtitle={selectedCheckpoint || "select a checkpoint"}
        >
          {meta.loading ? (
            <Loading />
          ) : meta.error ? (
            <ErrorBox error={meta.error} onRetry={meta.reload} />
          ) : historyRows.length ? (
            <>
              <LineChart
                series={[
                  {
                    label: "train loss",
                    values: historyRows.map((row: any) => row.train_loss ?? null),
                    color: "#64d8ff",
                  },
                  {
                    label: "val loss",
                    values: historyRows.map((row: any) => row.val_loss ?? null),
                    color: "#7cf5c8",
                  },
                ]}
                xLabels={chartX}
                height={190}
              />
              <div style={{ marginTop: 10 }}>
                <LineChart
                  series={[
                    {
                      label: "val accuracy",
                      values: historyRows.map((row: any) => row.val_accuracy ?? null),
                      color: "#f6c177",
                    },
                  ]}
                  xLabels={chartX}
                  height={150}
                  yFormat={(value) => `${(value * 100).toFixed(0)}%`}
                  zeroBase
                />
              </div>
            </>
          ) : meta.data ? (
            <KV
              rows={[
                ["checkpoint", meta.data.checkpoint_id],
                ["task", meta.data.task || "—"],
                ["format", meta.data.format || "—"],
                ["path", <span key="path" className="mono">{meta.data.path || "—"}</span>],
              ]}
            />
          ) : (
            <Empty>No history in this checkpoint.</Empty>
          )}
        </Panel>

        <Panel title="Checkpoint metadata" subtitle="raw JSON" flush>
          {meta.data ? (
            <pre style={{ border: "none", borderRadius: 0, maxHeight: 420 }}>
              {JSON.stringify(meta.data, null, 2)}
            </pre>
          ) : (
            <Empty>Select a checkpoint.</Empty>
          )}
        </Panel>
      </div>

      <div className="section-title">
        <h2>Artifact browser</h2>
        <span>experiment directory</span>
      </div>

      <div className="split">
        <Panel flush>
          {experiments.loading && !experiments.data ? (
            <Loading />
          ) : experiments.error ? (
            <ErrorBox error={experiments.error} onRetry={experiments.reload} />
          ) : experiments.data?.experiments?.length ? (
            <div className="table-wrap" style={{ border: "none" }}>
              <table>
                <thead>
                  <tr>
                    <th>file</th>
                    <th>kind</th>
                    <th className="num">size</th>
                    <th>modified</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {experiments.data.experiments.map((item) => (
                    <tr key={item.name}>
                      <td className="mono">{item.name}</td>
                      <td>
                        <Badge tone={item.kind === "benchmark_report" ? "cyan" : ""}>
                          {item.kind}
                        </Badge>
                      </td>
                      <td className="num mono">{fmtBytes(item.size_bytes)}</td>
                      <td className="mono">{fmtAgo(item.modified_at)}</td>
                      <td>
                        <Btn size="sm" onClick={() => setSelectedExperiment(item.name)}>
                          Open
                        </Btn>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty>No experiment files.</Empty>
          )}
        </Panel>

        <Panel
          title="Artifact detail"
          subtitle={selectedExperiment || "select a file"}
          flush
        >
          {!selectedExperiment ? (
            <Empty>Select an artifact.</Empty>
          ) : experimentDetail.loading && !experimentDetail.data ? (
            <Loading />
          ) : experimentDetail.error ? (
            <ErrorBox
              error={experimentDetail.error}
              onRetry={experimentDetail.reload}
            />
          ) : experimentDetail.data ? (
            <>
              <div style={{ padding: 14, borderBottom: "1px solid var(--line)" }}>
                <div className="row">
                  <Badge tone="mint">{experimentDetail.data.suite || "training"}</Badge>
                  <Badge>{experimentDetail.data.run_id || selectedExperiment}</Badge>
                  {experimentDetail.data.duration_seconds ? (
                    <Badge>
                      {fmtNum(experimentDetail.data.duration_seconds, 1)} s
                    </Badge>
                  ) : null}
                </div>
                {experimentDetail.data.summary ? (
                  <div style={{ marginTop: 10 }}>
                    <KV
                      rows={Object.entries(experimentDetail.data.summary)
                        .slice(0, 6)
                        .map(([key, value]) => [
                          key.replaceAll("_", " "),
                          typeof value === "number"
                            ? fmtNum(value, Math.abs(value) < 1 ? 4 : 2)
                            : typeof value === "object"
                              ? JSON.stringify(value)
                              : String(value),
                        ])}
                    />
                  </div>
                ) : null}
              </div>
              <pre style={{ border: "none", borderRadius: 0, maxHeight: 380 }}>
                {JSON.stringify(experimentDetail.data, null, 2)}
              </pre>
            </>
          ) : (
            <Empty>Artifact not loaded.</Empty>
          )}
        </Panel>
      </div>
    </>
  );
}
