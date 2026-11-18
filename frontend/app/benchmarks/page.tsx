"use client";

import { useEffect, useMemo, useState } from "react";

import { BarRows, LineChart } from "@/components/Charts";
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
  ProgressBar,
  StatusBadge,
} from "@/components/ui";
import {
  apiBlob,
  fmtAgo,
  fmtBytes,
  fmtCompact,
  fmtNum,
  fmtPct,
  post,
} from "@/lib/api";
import { useApi, useJob } from "@/lib/hooks";
import type { BenchmarkReportSummary, CheckpointMeta, Job } from "@/lib/types";

type Suite = {
  name: string;
  description: string;
};

type SuiteList = { suites: Suite[] };
type JobList = { jobs: Job[] };
type ReportList = { reports: BenchmarkReportSummary[] };

const COLUMNS: Record<string, Array<{ key: string; label: string; numeric?: boolean }>> = {
  reasoning: [
    { key: "mode", label: "mode" },
    { key: "max_steps", label: "steps", numeric: true },
    { key: "accuracy", label: "accuracy", numeric: true },
    { key: "loss", label: "loss", numeric: true },
    { key: "generation_mean_depth", label: "gen depth", numeric: true },
    { key: "generation_token_evaluations", label: "token evals", numeric: true },
    { key: "generation_estimated_macs", label: "est. MACs", numeric: true },
  ],
  adaptive_compute: [
    { key: "mode", label: "mode" },
    { key: "max_steps", label: "steps", numeric: true },
    { key: "accuracy", label: "accuracy", numeric: true },
    { key: "loss", label: "loss", numeric: true },
    { key: "generation_mean_depth", label: "gen depth", numeric: true },
    { key: "generation_token_evaluations", label: "token evals", numeric: true },
    { key: "generation_compute_vs_max_fixed_pct", label: "compute saved %", numeric: true },
    { key: "generation_speedup_vs_max_fixed", label: "speedup", numeric: true },
    { key: "generation_estimated_macs", label: "est. MACs", numeric: true },
  ],
  memory_recall: [
    { key: "adapt_round", label: "round", numeric: true },
    { key: "neural_recall_at_1", label: "neural R@1", numeric: true },
    { key: "lexical_recall_at_1", label: "lexical R@1", numeric: true },
    { key: "hybrid_recall_at_1", label: "hybrid R@1", numeric: true },
    { key: "state_updates", label: "state updates", numeric: true },
    { key: "state_bytes", label: "state bytes", numeric: true },
  ],
  inference: [
    { key: "mode", label: "mode" },
    { key: "max_steps", label: "steps", numeric: true },
    { key: "prompt_tokens", label: "prompt", numeric: true },
    { key: "latency_ms_p50", label: "p50 ms", numeric: true },
    { key: "latency_ms_p95", label: "p95 ms", numeric: true },
    { key: "tokens_per_second", label: "tok/s", numeric: true },
    { key: "mean_depth", label: "depth", numeric: true },
    { key: "token_evaluations", label: "token evals", numeric: true },
  ],
};

function cell(value: any, key: string): string {
  if (value === null || value === undefined) return "—";
  if (key.includes("accuracy") || key.includes("recall")) return fmtPct(value);
  if (key.includes("bytes")) return fmtBytes(value);
  if (key.includes("macs") || key.includes("token_evaluations")) return fmtCompact(value);
  if (key.includes("pct")) return `${fmtNum(value, 2)}%`;
  if (key.includes("seconds") || key.includes("latency")) return fmtNum(value, 2);
  if (key.includes("depth")) return fmtNum(value, 2);
  if (key.includes("speedup")) return `${fmtNum(value, 2)}×`;
  if (typeof value === "number") return fmtNum(value, 4);
  return String(value);
}

function rowsFor(report: any): any[] {
  if (!report) return [];
  if (report.suite === "ablation") {
    return report.results || [];
  }
  return report.results || [];
}

export default function BenchmarksPage() {
  const suites = useApi<SuiteList>("/api/benchmarks/suites", []);
  const jobs = useApi<JobList>("/api/benchmarks/jobs", []);
  const reports = useApi<ReportList>("/api/benchmarks/reports", []);
  const checkpoints = useApi<{ checkpoints: CheckpointMeta[] }>("/api/checkpoints", []);
  const [suite, setSuite] = useState("adaptive_compute");
  const [checkpointId, setCheckpointId] = useState("");
  const [limit, setLimit] = useState(150);
  const [batchSize, setBatchSize] = useState(16);
  const [maxNewTokens, setMaxNewTokens] = useState(12);
  const [maxExamples, setMaxExamples] = useState(5);
  const [depths, setDepths] = useState("1,2,3,4");
  const [modes, setModes] = useState("adaptive_soft,adaptive_hard");
  const [memoryRecords, setMemoryRecords] = useState(20);
  const [adaptRounds, setAdaptRounds] = useState(20);
  const [repeats, setRepeats] = useState(3);
  const [promptLengths, setPromptLengths] = useState("32,64,128");
  const [haltEpsilon, setHaltEpsilon] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { job, events, error: pollError } = useJob("benchmarks", jobId);

  const reportDetail = useApi<any>(
    selectedRun ? `/api/benchmarks/reports/${encodeURIComponent(selectedRun)}` : null,
    [selectedRun],
  );

  useEffect(() => {
    if (!selectedRun && reports.data?.reports?.length) {
      setSelectedRun(reports.data.reports[0].run_id);
    }
  }, [reports.data, selectedRun]);

  useEffect(() => {
    if (job?.status === "completed" && job.result?.run_id) {
      setSelectedRun(job.result.run_id);
      reports.reload();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.status, job?.result?.run_id]);

  const activeSuite = suites.data?.suites?.find((item) => item.name === suite);
  const detail = reportDetail.data;
  const columns = COLUMNS[detail?.suite] || COLUMNS.adaptive_compute;
  const rows = rowsFor(detail);
  const memoryRows = detail?.memory_recall || [];

  const accuracyRows = useMemo(
    () =>
      rows
        .filter((row) => typeof row.accuracy === "number")
        .map((row) => ({
          label: `${row.mode}@${row.max_steps ?? "?"}`,
          value: row.accuracy * 100,
        })),
    [rows],
  );
  const computeRows = useMemo(
    () =>
      rows
        .filter((row) => typeof row.generation_token_evaluations === "number")
        .map((row) => ({
          label: `${row.mode}@${row.max_steps ?? "?"}`,
          value: row.generation_token_evaluations,
        })),
    [rows],
  );
  const memorySeries = useMemo(
    () =>
      detail?.suite === "memory_recall"
        ? [
            {
              label: "neural R@1",
              values: rows.map((row) => row.neural_recall_at_1 ?? null),
              color: "#7cf5c8",
            },
            {
              label: "lexical R@1",
              values: rows.map((row) => row.lexical_recall_at_1 ?? null),
              color: "#64d8ff",
            },
            {
              label: "hybrid R@1",
              values: rows.map((row) => row.hybrid_recall_at_1 ?? null),
              color: "#f6c177",
            },
          ]
        : [],
    [detail?.suite, rows],
  );

  async function start() {
    setStarting(true);
    setError(null);
    try {
      const options: Record<string, any> = {
        limit,
        batch_size: batchSize,
        max_new_tokens: maxNewTokens,
        max_examples: maxExamples,
        device: "auto",
      };
      if (depths.trim()) {
        options.depths = depths
          .split(",")
          .map((value) => Number(value.trim()))
          .filter((value) => Number.isFinite(value) && value > 0);
      }
      if (modes.trim()) {
        options.modes = modes
          .split(",")
          .map((value) => value.trim())
          .filter(Boolean);
      }
      if (suite === "memory_recall" || suite === "ablation") {
        options.memory_records = memoryRecords;
        options.adapt_rounds = adaptRounds;
      }
      if (suite === "inference") {
        options.repeats = repeats;
        options.prompt_lengths = promptLengths
          .split(",")
          .map((value) => Number(value.trim()))
          .filter((value) => Number.isFinite(value) && value > 0);
      }
      if (haltEpsilon.trim()) options.halt_epsilon = Number(haltEpsilon);
      const response = await post<{ job_id: string }>("/api/benchmarks/start", {
        suite,
        checkpoint_id: checkpointId || null,
        options,
      });
      setJobId(response.job_id);
      jobs.reload();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setStarting(false);
    }
  }

  async function cancel(id: string) {
    setError(null);
    try {
      await post(`/api/benchmarks/jobs/${id}/cancel`, {});
      jobs.reload();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function downloadArtifact(format: "csv" | "json") {
    if (!selectedRun) return;
    setError(null);
    try {
      const blob = await apiBlob(
        `/api/benchmarks/reports/${encodeURIComponent(selectedRun)}/${format}`,
      );
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${selectedRun}.${format}`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <>
      <PageHeader
        kicker="WILL / BENCHMARKS"
        title="Measure compute, not vibes."
        description="Every suite executes the model on held-out or synthetic data. Accuracy, depth and token evaluations are measured; MAC/FLOP values are analytic estimates. Wall-clock overhead from hard early exit is reported honestly."
        actions={
          <>
            <Badge tone="mint">JSON + CSV</Badge>
            <Btn onClick={() => reports.reload()}>Refresh reports</Btn>
          </>
        }
      />

      {error ? <ErrorBox error={error} /> : null}

      <div className="split">
        <Panel
          title="Run a suite"
          subtitle={activeSuite?.description || "Select a suite"}
          actions={
            <Btn variant="primary" onClick={start} disabled={starting}>
              {starting ? "Queueing…" : "Start benchmark"}
            </Btn>
          }
        >
          <div className="form-grid">
            <Field label="Suite">
              <select value={suite} onChange={(e) => setSuite(e.target.value)}>
                {(suites.data?.suites || []).map((item) => (
                  <option key={item.name} value={item.name}>
                    {item.name}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Checkpoint" hint="blank = current default checkpoint">
              <select value={checkpointId} onChange={(e) => setCheckpointId(e.target.value)}>
                <option value="">current default</option>
                {checkpoints.data?.checkpoints?.map((ckpt) => (
                  <option key={ckpt.checkpoint_id} value={ckpt.checkpoint_id}>
                    {ckpt.checkpoint_id}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Eval limit" hint="examples from the split">
              <input
                type="number"
                min={1}
                value={limit}
                onChange={(e) => setLimit(Number(e.target.value))}
              />
            </Field>
            <Field label="Batch size">
              <input
                type="number"
                min={1}
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
              />
            </Field>
            <Field label="Max new tokens">
              <input
                type="number"
                min={1}
                value={maxNewTokens}
                onChange={(e) => setMaxNewTokens(Number(e.target.value))}
              />
            </Field>
            <Field label="Example rows shown">
              <input
                type="number"
                min={1}
                value={maxExamples}
                onChange={(e) => setMaxExamples(Number(e.target.value))}
              />
            </Field>
            <Field label="Depths" hint="comma-separated fixed depths">
              <input value={depths} onChange={(e) => setDepths(e.target.value)} />
            </Field>
            <Field label="Modes" hint="adaptive_soft, adaptive_hard">
              <input value={modes} onChange={(e) => setModes(e.target.value)} />
            </Field>
            <Field label="Halt epsilon" hint="blank = checkpoint default">
              <input
                value={haltEpsilon}
                onChange={(e) => setHaltEpsilon(e.target.value)}
                placeholder="0.05"
              />
            </Field>
            {suite === "memory_recall" || suite === "ablation" ? (
              <>
                <Field label="Memory records">
                  <input
                    type="number"
                    min={2}
                    value={memoryRecords}
                    onChange={(e) => setMemoryRecords(Number(e.target.value))}
                  />
                </Field>
                <Field label="Adapt rounds">
                  <input
                    type="number"
                    min={1}
                    max={200}
                    value={adaptRounds}
                    onChange={(e) => setAdaptRounds(Number(e.target.value))}
                  />
                </Field>
              </>
            ) : null}
            {suite === "inference" ? (
              <>
                <Field label="Repeats">
                  <input
                    type="number"
                    min={1}
                    value={repeats}
                    onChange={(e) => setRepeats(Number(e.target.value))}
                  />
                </Field>
                <Field label="Prompt lengths">
                  <input
                    value={promptLengths}
                    onChange={(e) => setPromptLengths(e.target.value)}
                  />
                </Field>
              </>
            ) : null}
          </div>
          <div className="notice" style={{ marginTop: 14 }}>
            <strong>{suite}</strong> writes <span className="mono">report.json</span> and{" "}
            <span className="mono">summary.csv</span> under the experiment directory.
            Hard early exit can reduce token evaluations while adding gather/scatter
            overhead on CPU; the report shows both.
          </div>
        </Panel>

        <div className="stack">
          <Panel
            title="Active benchmark"
            subtitle={jobId || "no job selected"}
            actions={job ? <StatusBadge status={job.status} /> : null}
          >
            {!jobId ? (
              <Empty>Start a suite or open a recent job.</Empty>
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
                  <Badge tone="cyan">{job.config?.suite || "—"}</Badge>
                  <Badge>{job.config?.checkpoint_id || "default"}</Badge>
                  {job.status === "running" || job.status === "queued" ? (
                    <Btn size="sm" variant="danger" onClick={() => cancel(job.id)}>
                      Cancel
                    </Btn>
                  ) : null}
                </div>
                {job.result ? (
                  <div className="kv" style={{ marginTop: 14 }}>
                    <div className="k">run id</div>
                    <div className="v">{job.result.run_id}</div>
                    <div className="k">suite</div>
                    <div className="v">{job.result.suite}</div>
                    <div className="k">duration</div>
                    <div className="v">{fmtNum(job.result.duration_seconds, 1)} s</div>
                    <div className="k">rows</div>
                    <div className="v">{job.result.result_count}</div>
                  </div>
                ) : null}
                {job.error ? (
                  <div className="error-box" style={{ marginTop: 12, whiteSpace: "pre-wrap" }}>
                    {job.error}
                  </div>
                ) : null}
              </>
            )}
          </Panel>

          <Panel title="Events" subtitle="persisted benchmark progress">
            <div className="log" style={{ maxHeight: 210 }}>
              {events.length ? (
                events.slice(-60).map((event, index) => (
                  <div className="log-line" key={`${event._event_id ?? index}-${event.type}`}>
                    <span className="t">{event.type}</span>
                    <span>
                      {event.label ||
                        event.message ||
                        `${event.suite || ""} ${event.completed ?? ""}/${event.total ?? ""}`}
                    </span>
                  </div>
                ))
              ) : (
                <div className="empty">No events yet.</div>
              )}
            </div>
          </Panel>
        </div>
      </div>

      <div className="section-title">
        <h2>Reports</h2>
        <span>{reports.data?.reports?.length ?? 0} on disk</span>
      </div>

      <div className="split">
        <Panel flush>
          {reports.loading && !reports.data ? (
            <Loading />
          ) : reports.error ? (
            <ErrorBox error={reports.error} onRetry={reports.reload} />
          ) : reports.data?.reports?.length ? (
            <div className="table-wrap" style={{ border: "none" }}>
              <table>
                <thead>
                  <tr>
                    <th>suite</th>
                    <th>checkpoint</th>
                    <th className="num">duration</th>
                    <th>device</th>
                    <th>created</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {reports.data.reports.map((report) => (
                    <tr key={report.run_id}>
                      <td className="mono">{report.suite}</td>
                      <td>{report.checkpoint_id || "fresh"}</td>
                      <td className="num">{fmtNum(report.duration_seconds, 1)} s</td>
                      <td className="mono">{report.device}</td>
                      <td className="mono">{fmtAgo(report.created_at)}</td>
                      <td>
                        <Btn size="sm" onClick={() => setSelectedRun(report.run_id)}>
                          Inspect
                        </Btn>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty>No reports yet.</Empty>
          )}
        </Panel>

        <Panel
          title={detail?.suite || "Report detail"}
          subtitle={selectedRun || "select a report"}
          actions={
            detail?.artifacts ? (
              <>
                <Btn size="sm" onClick={() => downloadArtifact("csv")}>
                  CSV
                </Btn>
                <Btn size="sm" onClick={() => downloadArtifact("json")}>
                  JSON
                </Btn>
              </>
            ) : null
          }
        >
          {!selectedRun ? (
            <Empty>Select a report.</Empty>
          ) : reportDetail.loading && !detail ? (
            <Loading />
          ) : reportDetail.error ? (
            <ErrorBox error={reportDetail.error} onRetry={reportDetail.reload} />
          ) : detail ? (
            <>
              <KV
                rows={Object.entries(detail.summary || {})
                  .slice(0, 8)
                  .map(([key, value]) => [
                    key.replaceAll("_", " "),
                    typeof value === "number"
                      ? fmtNum(value, Math.abs(value) < 1 ? 4 : 2)
                      : typeof value === "object"
                        ? JSON.stringify(value)
                        : String(value),
                  ])}
              />
              {detail.suite === "memory_recall" && memorySeries.length ? (
                <div style={{ marginTop: 16 }}>
                  <LineChart
                    series={memorySeries}
                    xLabels={rows.map((row) => `r${row.adapt_round}`)}
                    height={180}
                    yFormat={(value) => `${(value * 100).toFixed(0)}%`}
                    zeroBase
                  />
                </div>
              ) : null}
              {detail.suite !== "memory_recall" && detail.suite !== "inference" && accuracyRows.length ? (
                <div className="grid grid-2" style={{ marginTop: 16 }}>
                  <div>
                    <div className="stat-label" style={{ marginBottom: 8 }}>
                      accuracy by configuration
                    </div>
                    <BarRows rows={accuracyRows} format={(value) => `${value.toFixed(1)}%`} />
                  </div>
                  <div>
                    <div className="stat-label" style={{ marginBottom: 8 }}>
                      generation token evaluations
                    </div>
                    <BarRows rows={computeRows} color="#64d8ff" />
                  </div>
                </div>
              ) : null}
              {detail.suite === "inference" && rows.length ? (
                <div style={{ marginTop: 16 }}>
                  <div className="stat-label" style={{ marginBottom: 8 }}>
                    p50 latency by configuration
                  </div>
                  <BarRows
                    rows={rows.map((row) => ({
                      label: `${row.mode}@${row.max_steps} len=${row.prompt_tokens}`,
                      value: row.latency_ms_p50,
                    }))}
                    color="#f6c177"
                    format={(value) => `${value.toFixed(1)} ms`}
                  />
                </div>
              ) : null}
              <div style={{ marginTop: 16 }}>
                <div className="stat-label" style={{ marginBottom: 8 }}>
                  measured rows
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        {columns.map((column) => (
                          <th className={column.numeric ? "num" : ""} key={column.key}>
                            {column.label}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row, index) => (
                        <tr key={`${row.mode || row.adapt_round || index}-${index}`}>
                          {columns.map((column) => (
                            <td
                              className={column.numeric ? "num mono" : column.key === "mode" ? "mono" : ""}
                              key={column.key}
                            >
                              {cell(row[column.key], column.key)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              {detail.suite === "ablation" && memoryRows.length ? (
                <div style={{ marginTop: 16 }}>
                  <div className="stat-label" style={{ marginBottom: 8 }}>
                    memory subsystem
                  </div>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          {COLUMNS.memory_recall.map((column) => (
                            <th className={column.numeric ? "num" : ""} key={column.key}>
                              {column.label}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {memoryRows.map((row: any, index: number) => (
                          <tr key={`${row.adapt_round}-${index}`}>
                            {COLUMNS.memory_recall.map((column) => (
                              <td className={column.numeric ? "num mono" : ""} key={column.key}>
                                {cell(row[column.key], column.key)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : null}
              <div className="row" style={{ marginTop: 12 }}>
                {(detail.notes || []).map((note: string) => (
                  <Badge key={note} tone="amber">{note}</Badge>
                ))}
              </div>
              <details style={{ marginTop: 14 }}>
                <summary className="mono" style={{ color: "var(--muted)", cursor: "pointer", fontSize: 10 }}>
                  RAW REPORT JSON
                </summary>
                <pre style={{ marginTop: 8, maxHeight: 300 }}>
                  {JSON.stringify(detail, null, 2)}
                </pre>
              </details>
            </>
          ) : (
            <Empty>Report not loaded.</Empty>
          )}
        </Panel>
      </div>

      <div className="section-title">
        <h2>Recent benchmark jobs</h2>
        <span>SQLite-backed</span>
      </div>
      <Panel flush>
        {jobs.data?.jobs?.length ? (
          <div className="table-wrap" style={{ border: "none" }}>
            <table>
              <thead>
                <tr>
                  <th>job</th>
                  <th>suite</th>
                  <th>status</th>
                  <th>checkpoint</th>
                  <th className="num">rows</th>
                  <th>run</th>
                  <th>created</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {jobs.data.jobs.map((row) => (
                  <tr key={row.id}>
                    <td className="mono">{row.id.slice(0, 10)}</td>
                    <td className="mono">{row.config?.suite || "—"}</td>
                    <td><StatusBadge status={row.status} /></td>
                    <td>{row.config?.checkpoint_id || "default"}</td>
                    <td className="num">{row.result?.result_count ?? "—"}</td>
                    <td className="mono">{row.result?.run_id || "—"}</td>
                    <td className="mono">{fmtAgo(row.created_at)}</td>
                    <td>
                      <div className="row" style={{ flexWrap: "nowrap" }}>
                        <Btn size="sm" onClick={() => setJobId(row.id)}>Open</Btn>
                        {row.status === "running" || row.status === "queued" ? (
                          <Btn size="sm" variant="danger" onClick={() => cancel(row.id)}>
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
          <Empty>No benchmark jobs yet.</Empty>
        )}
      </Panel>
    </>
  );
}
