"use client";

import Link from "next/link";

import { BarRows } from "@/components/Charts";
import {
  Badge,
  Empty,
  ErrorBox,
  KV,
  Loading,
  PageHeader,
  Panel,
  Stat,
  StatusBadge,
} from "@/components/ui";
import { fmtAgo, fmtNum, fmtPct } from "@/lib/api";
import { useApi } from "@/lib/hooks";
import type { BenchmarkReportSummary, CheckpointMeta } from "@/lib/types";

type Overview = {
  model: {
    default_checkpoint_id: string | null;
    device: string;
    latest_checkpoint: CheckpointMeta | null;
    checkpoint_count: number;
    parameters_total: number | null;
  };
  counts: {
    checkpoints: number;
    memory_sessions: number;
    jobs_total: number;
    jobs_running: number;
    benchmark_reports: number;
  };
  recent_jobs: Array<Record<string, any>>;
  recent_reports: BenchmarkReportSummary[];
  recent_sessions: Array<Record<string, any>>;
  recent_checkpoints: CheckpointMeta[];
};

export default function OverviewPage() {
  const overview = useApi<Overview>("/api/overview", []);
  const system = useApi<Record<string, any>>("/api/system", []);
  const current = useApi<Record<string, any>>("/api/checkpoints/current", []);
  const reports = useApi<{ reports: BenchmarkReportSummary[] }>(
    "/api/benchmarks/reports",
    [],
  );

  const latestReport = reports.data?.reports?.[0];
  const reportDetail = useApi<Record<string, any>>(
    latestReport ? `/api/benchmarks/reports/${latestReport.run_id}` : null,
    [latestReport?.run_id],
  );

  if (overview.error) {
    return (
      <>
        <PageHeader kicker="WILL / OVERVIEW" title="Adaptive computation, measured." />
        <ErrorBox error={overview.error} onRetry={overview.reload} />
      </>
    );
  }

  const model = overview.data?.model;
  const latest = model?.latest_checkpoint;
  const latestMetrics = latest?.metrics || {};
  const bestAcc =
    latestMetrics.best_val_accuracy ?? latestMetrics.val_accuracy ?? null;
  const bestLoss =
    latestMetrics.best_val_loss ?? latestMetrics.val_loss ?? null;
  const params =
    model?.parameters_total ??
    current.data?.parameters_total ??
    latestMetrics.parameters_total ??
    null;

  return (
    <>
      <PageHeader
        kicker="WILL / OVERVIEW"
        title="Adaptive computation, measured."
        description="A working research system for recurrent depth allocation and session-local fast weights. Every number on this console is read from a live API backed by PyTorch execution, checkpoints and measured jobs."
        actions={
          <>
            <Link href="/training" className="btn primary">
              Start a training run
            </Link>
            <Link href="/benchmarks" className="btn">
              Run a benchmark
            </Link>
          </>
        }
      />

      {overview.loading && !overview.data ? (
        <Loading label="Reading laboratory state" />
      ) : null}

      <div className="stat-grid">
        <Stat
          label="Slow parameters"
          value={fmtNum(params)}
          small="weights"
          foot={latest ? `latest: ${latest.checkpoint_id}` : "no checkpoint yet"}
          tone="mint"
        />
        <Stat
          label="Best validation accuracy"
          value={bestAcc === null ? "—" : fmtPct(bestAcc)}
          small={bestLoss !== null ? `loss ${bestLoss.toFixed(4)}` : undefined}
          foot="exact-match generation on held-out data"
          tone="cyan"
        />
        <Stat
          label="Checkpoints"
          value={fmtNum(model?.checkpoint_count ?? overview.data?.counts.checkpoints)}
          small="on disk"
          foot={`${overview.data?.counts.jobs_running ?? 0} jobs running`}
        />
        <Stat
          label="Memory sessions"
          value={fmtNum(overview.data?.counts.memory_sessions)}
          small="isolated"
          foot={`${overview.data?.counts.benchmark_reports ?? 0} benchmark reports`}
          tone="amber"
        />
      </div>

      <div className="section-title">
        <h2>Latest measured run</h2>
        <span>real report data</span>
      </div>

      <div className="split">
        <Panel
          title={latestReport ? latestReport.suite : "No benchmark report"}
          subtitle={
            latestReport
              ? `${latestReport.run_id} · ${fmtAgo(latestReport.created_at)} · ${latestReport.device}`
              : "Run a benchmark suite to populate this panel."
          }
          actions={
            latestReport ? (
              <Link href="/benchmarks" className="btn sm">
                Inspect reports
              </Link>
            ) : null
          }
        >
          {latestReport ? (
            <>
              <div className="row" style={{ marginBottom: 14 }}>
                <Badge tone="mint">{latestReport.suite}</Badge>
                <Badge>{fmtNum(latestReport.duration_seconds, 1)} s</Badge>
                <Badge>
                  {latestReport.checkpoint_id || "fresh model"}
                </Badge>
              </div>
              <div className="grid grid-2">
                <KV
                  rows={Object.entries(latestReport.summary || {})
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
                <div>
                  <div className="stat-label" style={{ marginBottom: 8 }}>
                    result rows
                  </div>
                  {reportDetail.data?.results?.length ? (
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>mode</th>
                            <th className="num">steps</th>
                            <th className="num">acc</th>
                            <th className="num">depth</th>
                          </tr>
                        </thead>
                        <tbody>
                          {reportDetail.data.results.slice(0, 5).map((row: any, index: number) => (
                            <tr key={`${row.mode}-${row.max_steps}-${index}`}>
                              <td className="mono">{row.mode || row.adapt_round || "—"}</td>
                              <td className="num">{row.max_steps ?? row.adapt_round ?? "—"}</td>
                              <td className="num">
                                {typeof row.accuracy === "number"
                                  ? fmtPct(row.accuracy)
                                  : typeof row.neural_recall_at_1 === "number"
                                    ? fmtPct(row.neural_recall_at_1)
                                    : "—"}
                              </td>
                              <td className="num">
                                {typeof row.generation_mean_depth === "number"
                                  ? row.generation_mean_depth.toFixed(2)
                                  : typeof row.mean_depth === "number"
                                    ? row.mean_depth.toFixed(2)
                                    : "—"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <Empty>Report detail is loading or has no tabular rows.</Empty>
                  )}
                </div>
              </div>
            </>
          ) : (
            <Empty>No reports yet. The benchmark page runs real suites.</Empty>
          )}
        </Panel>

        <Panel
          title="Execution environment"
          subtitle="reported by the API process"
          actions={<Badge tone={system.data?.cuda_available ? "mint" : "amber"}>{system.data?.cuda_available ? "CUDA" : "CPU"}</Badge>}
        >
          {system.loading ? (
            <Loading />
          ) : system.error ? (
            <ErrorBox error={system.error} onRetry={system.reload} />
          ) : (
            <KV
              rows={[
                ["platform", system.data?.platform || "—"],
                ["python", system.data?.python || "—"],
                ["torch", system.data?.torch || "—"],
                ["device", model?.device || "—"],
                ["cpu threads", String(system.data?.cpu_threads ?? "—")],
                ["process rss", `${fmtNum(system.data?.process_rss_mb, 1)} MB`],
                ["checkpoint dir", <span key="ckpt" className="mono">{system.data?.checkpoint_dir || "—"}</span>],
              ]}
            />
          )}
        </Panel>
      </div>

      <div className="section-title">
        <h2>Recent activity</h2>
        <span>jobs, checkpoints, sessions</span>
      </div>

      <div className="grid grid-2">
        <Panel
          title="Recent jobs"
          subtitle="training and benchmark workers"
          actions={<Link href="/training" className="btn sm">Training</Link>}
          flush
        >
          {overview.data?.recent_jobs?.length ? (
            <div className="table-wrap" style={{ border: "none" }}>
              <table>
                <thead>
                  <tr>
                    <th>kind</th>
                    <th>status</th>
                    <th>message</th>
                    <th>created</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.data.recent_jobs.slice(0, 6).map((job) => (
                    <tr key={job.id}>
                      <td className="mono">{job.kind}</td>
                      <td><StatusBadge status={job.status} /></td>
                      <td>{job.message || "—"}</td>
                      <td className="mono">{fmtAgo(job.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty>No jobs have been run yet.</Empty>
          )}
        </Panel>

        <Panel
          title="Recent checkpoints"
          subtitle="per-task saved slow weights"
          actions={<Link href="/experiments" className="btn sm">All checkpoints</Link>}
          flush
        >
          {overview.data?.recent_checkpoints?.length ? (
            <div className="table-wrap" style={{ border: "none" }}>
              <table>
                <thead>
                  <tr>
                    <th>checkpoint</th>
                    <th>task</th>
                    <th className="num">best acc</th>
                    <th>created</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.data.recent_checkpoints.slice(0, 6).map((ckpt) => {
                    const metrics = ckpt.metrics || {};
                    const acc = metrics.best_val_accuracy ?? metrics.val_accuracy;
                    return (
                      <tr key={ckpt.checkpoint_id}>
                        <td className="mono">{ckpt.checkpoint_id}</td>
                        <td>{ckpt.task || "—"}</td>
                        <td className="num">{typeof acc === "number" ? fmtPct(acc) : "—"}</td>
                        <td className="mono">{fmtAgo(ckpt.created_at)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty>No checkpoints yet.</Empty>
          )}
        </Panel>
      </div>

      <div className="grid grid-2" style={{ marginTop: 16 }}>
        <Panel title="Memory sessions" subtitle="session-local fast weights" flush>
          {overview.data?.recent_sessions?.length ? (
            <div className="table-wrap" style={{ border: "none" }}>
              <table>
                <thead>
                  <tr>
                    <th>name</th>
                    <th className="num">chunks</th>
                    <th className="num">updates</th>
                    <th>created</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.data.recent_sessions.slice(0, 6).map((session) => (
                    <tr key={session.session_id}>
                      <td className="mono">{session.name}</td>
                      <td className="num">{session.chunk_count ?? session.chunks?.length ?? 0}</td>
                      <td className="num">{session.state_updates ?? 0}</td>
                      <td className="mono">{fmtAgo(session.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty>No memory sessions yet.</Empty>
          )}
        </Panel>

        <Panel title="Benchmark reports" subtitle="JSON + CSV artifacts on disk" flush>
          {reports.data?.reports?.length ? (
            <div className="table-wrap" style={{ border: "none" }}>
              <table>
                <thead>
                  <tr>
                    <th>suite</th>
                    <th>checkpoint</th>
                    <th className="num">duration</th>
                    <th>created</th>
                  </tr>
                </thead>
                <tbody>
                  {reports.data.reports.slice(0, 6).map((report) => (
                    <tr key={report.run_id}>
                      <td className="mono">{report.suite}</td>
                      <td>{report.checkpoint_id || "fresh"}</td>
                      <td className="num">{fmtNum(report.duration_seconds, 1)} s</td>
                      <td className="mono">{fmtAgo(report.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty>No benchmark reports yet.</Empty>
          )}
        </Panel>
      </div>

      <div style={{ marginTop: 18 }}>
        <Panel title="Compute ledger" tight>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span className="panel-sub">
              MAC/FLOP values are analytic estimates from tensor shapes. Latency and
              token evaluations are measured. No energy savings are claimed without a
              power measurement.
            </span>
            <Badge tone="amber">estimates labelled</Badge>
          </div>
        </Panel>
      </div>
    </>
  );
}
