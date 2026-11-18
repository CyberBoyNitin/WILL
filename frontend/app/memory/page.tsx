"use client";

import { useEffect, useMemo, useRef, useState } from "react";

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
  post,
} from "@/lib/api";
import { useApi } from "@/lib/hooks";
import type { MemorySessionSummary } from "@/lib/types";

type QueryResult = {
  question: string;
  answer: string;
  answer_type: string;
  grounded: boolean;
  insufficient: boolean;
  confidence: number;
  sources: Array<Record<string, any>>;
  mode: string;
  latency_ms: number;
  notes: string;
  memory_used_in_generation: boolean;
};

type SearchResult = {
  question: string;
  mode: string;
  results: Array<Record<string, any>>;
};

export default function MemoryPage() {
  const sessions = useApi<{ sessions: MemorySessionSummary[] }>(
    "/api/memory/sessions",
    [],
  );
  const [selected, setSelected] = useState<string | null>(null);
  const detail = useApi<MemorySessionSummary>(
    selected ? `/api/memory/sessions/${selected}` : null,
    [selected],
  );
  const [createName, setCreateName] = useState("research notes");
  const [ingestText, setIngestText] = useState(
    "K00=V100. K01=V101. K02=V102. access code alpha_1234.",
  );
  const [ingestName, setIngestName] = useState("inline.txt");
  const [question, setQuestion] = useState("what is the value of K00?");
  const [mode, setMode] = useState<"lexical" | "neural" | "hybrid">("hybrid");
  const [topK, setTopK] = useState(5);
  const [useMemory, setUseMemory] = useState(true);
  const [adaptRounds, setAdaptRounds] = useState(5);
  const [answer, setAnswer] = useState<QueryResult | null>(null);
  const [search, setSearch] = useState<SearchResult | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const justDeleted = useRef<string | null>(null);

  useEffect(() => {
    if (selected) return;
    const available = sessions.data?.sessions || [];
    const next = available.find(
      (session) => session.session_id !== justDeleted.current,
    );
    if (next) setSelected(next.session_id);
  }, [sessions.data, selected]);

  const active = detail.data;
  const documents = active?.documents || [];
  const chunks = active?.chunks || [];

  const sessionOptions = useMemo(
    () => sessions.data?.sessions || [],
    [sessions.data],
  );

  function refresh() {
    sessions.reload();
    detail.reload();
  }

  async function run(label: string, fn: () => Promise<void>) {
    setBusy(label);
    setError(null);
    setMessage(null);
    try {
      await fn();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function createSession() {
    await run("create", async () => {
      const created = await post<MemorySessionSummary>("/api/memory/sessions", {
        name: createName,
        metadata: {},
      });
      justDeleted.current = null;
      setSelected(created.session_id);
      setAnswer(null);
      setSearch(null);
      sessions.reload();
      setMessage(`Created session ${created.session_id.slice(0, 10)}.`);
    });
  }

  async function ingestPaste() {
    if (!selected) return;
    await run("ingest", async () => {
      const result = await post<any>(
        `/api/memory/sessions/${selected}/ingest-text`,
        { text: ingestText, filename: ingestName || "inline.txt" },
      );
      setMessage(
        `Ingested ${result.ingest?.chunks_added ?? 0} chunk(s); ${result.ingest?.updates ?? 0} state update(s).`,
      );
      refresh();
    });
  }

  async function ingestFile() {
    const file = fileRef.current?.files?.[0];
    if (!file || !selected) return;
    await run("upload", async () => {
      const body = new FormData();
      body.append("file", file);
      const result = await api<any>(`/api/memory/sessions/${selected}/ingest`, {
        method: "POST",
        body,
      });
      setMessage(
        `Ingested ${result.ingest?.chunks_added ?? 0} chunk(s) from ${file.name}.`,
      );
      if (fileRef.current) fileRef.current.value = "";
      refresh();
    });
  }

  async function ask() {
    if (!selected || !question.trim()) return;
    await run("query", async () => {
      const result = await post<QueryResult>(
        `/api/memory/sessions/${selected}/query`,
        {
          question,
          mode,
          top_k: topK,
          use_memory_in_generation: useMemory,
        },
      );
      setAnswer(result);
      detail.reload();
    });
  }

  async function runSearch() {
    if (!selected || !question.trim()) return;
    await run("search", async () => {
      const result = await post<SearchResult>(
        `/api/memory/sessions/${selected}/search`,
        { question, mode, top_k: topK },
      );
      setSearch(result);
    });
  }

  async function adapt() {
    if (!selected) return;
    await run("adapt", async () => {
      const result = await post<any>(
        `/api/memory/sessions/${selected}/adapt`,
        { rounds: adaptRounds },
      );
      setMessage(
        result.message
          ? result.message
          : `Adapted ${result.updates ?? 0} update(s) across ${result.rounds ?? 0} round(s); ${result.total_updates ?? 0} total state update(s) in ${fmtNum(result.adapt_seconds, 2)} s.`,
      );
      refresh();
    });
  }

  async function sessionAction(action: "save" | "restore" | "reset") {
    if (!selected) return;
    await run(action, async () => {
      const result = await post<any>(
        `/api/memory/sessions/${selected}/${action}`,
        {},
      );
      setMessage(`${action} completed: ${result.session_id || selected}.`);
      refresh();
    });
  }

  async function removeSession() {
    const target = selected;
    if (!target) return;
    setBusy("delete");
    setError(null);
    setMessage(null);
    justDeleted.current = target;
    setSelected(null);
    setAnswer(null);
    setSearch(null);
    try {
      await api(`/api/memory/sessions/${target}`, { method: "DELETE" });
      sessions.reload();
      setMessage("Session deleted.");
    } catch (err) {
      setSelected(target);
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function removeDocument(docId: string) {
    if (!selected) return;
    await run("delete-doc", async () => {
      await api(`/api/memory/sessions/${selected}/documents/${docId}`, {
        method: "DELETE",
      });
      setMessage("Document removed.");
      refresh();
    });
  }

  return (
    <>
      <PageHeader
        kicker="WILL / MEMORY"
        title="A session that can learn, then forget."
        description="Fast weights adapt inside a UUID-isolated session. Slow parameters stay frozen. Retrieval is lexical, neural or hybrid, and answers cite the source chunks or explicitly say the information is insufficient."
        actions={
          <>
            <Badge tone="mint">session-isolated</Badge>
            <Badge tone="cyan">bounded state</Badge>
            <Btn onClick={refresh}>Refresh</Btn>
          </>
        }
      />

      {error ? <ErrorBox error={error} /> : null}
      {message ? (
        <div className="notice" style={{ marginBottom: 14 }}>
          {message}
        </div>
      ) : null}

      <div className="split">
        <div className="stack">
          <Panel
            title="Sessions"
            subtitle="each UUID owns a separate FastMemoryState"
            actions={
              <Badge>{sessions.data?.sessions?.length ?? 0} total</Badge>
            }
            flush
          >
            {sessions.loading && !sessions.data ? (
              <Loading />
            ) : sessions.error ? (
              <ErrorBox error={sessions.error} onRetry={sessions.reload} />
            ) : sessionOptions.length ? (
              <div className="table-wrap" style={{ border: "none" }}>
                <table>
                  <thead>
                    <tr>
                      <th>name</th>
                      <th>session</th>
                      <th className="num">chunks</th>
                      <th className="num">updates</th>
                      <th>created</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {sessionOptions.map((session) => (
                      <tr key={session.session_id}>
                        <td className="mono">{session.name}</td>
                        <td className="mono">{session.session_id.slice(0, 10)}</td>
                        <td className="num">{session.chunk_count ?? 0}</td>
                        <td className="num">{session.state_updates ?? 0}</td>
                        <td className="mono">{fmtAgo(session.created_at)}</td>
                        <td>
                          <Btn
                            size="sm"
                            onClick={() => {
                              justDeleted.current = null;
                              setSelected(session.session_id);
                              setAnswer(null);
                              setSearch(null);
                            }}
                          >
                            {selected === session.session_id ? "Open" : "Select"}
                          </Btn>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <Empty>No sessions yet. Create one below.</Empty>
            )}
            <div
              className="row"
              style={{ padding: 14, borderTop: "1px solid var(--line)" }}
            >
              <input
                value={createName}
                onChange={(event) => setCreateName(event.target.value)}
                placeholder="session name"
                style={{
                  flex: 1,
                  minWidth: 160,
                  padding: "8px 10px",
                  border: "1px solid var(--line)",
                  borderRadius: 4,
                  background: "#070b10",
                  color: "var(--text)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 12,
                }}
              />
              <Btn
                variant="primary"
                onClick={createSession}
                disabled={busy === "create" || !createName.trim()}
              >
                {busy === "create" ? "Creating…" : "New session"}
              </Btn>
            </div>
          </Panel>

          {active ? (
            <>
              <div className="stat-grid">
                <Stat
                  label="Chunks"
                  value={fmtNum(active.chunk_count)}
                  small={`/ ${active.capacity_chunks ?? "—"}`}
                  foot={`${active.record_count ?? 0} extracted records`}
                  tone="mint"
                />
                <Stat
                  label="State updates"
                  value={fmtNum(active.state_updates)}
                  small="adaptations"
                  foot={`${fmtNum(active.fast_parameters)} fast parameters`}
                  tone="cyan"
                />
                <Stat
                  label="Fast state"
                  value={fmtBytes(active.memory_state_bytes)}
                  foot={`${fmtBytes(active.total_state_bytes)} total on disk`}
                  tone="amber"
                />
                <Stat
                  label="Adapt time"
                  value={fmtNum(active.adapt_seconds, 2)}
                  small="s"
                  foot={`${active.memory_kind} · dim ${active.memory_dim ?? "—"}`}
                />
              </div>

              <Panel
                title="Ingest"
                subtitle="PDF, TXT or Markdown; text is chunked and encoded into fast weights"
              >
                <div className="form-grid">
                  <Field label="Inline filename">
                    <input
                      value={ingestName}
                      onChange={(event) => setIngestName(event.target.value)}
                    />
                  </Field>
                  <Field label="Upload document" hint="server enforces the configured byte limit">
                    <input ref={fileRef} type="file" accept=".pdf,.txt,.md,.markdown" />
                  </Field>
                  <Field label="File action">
                    <Btn
                      onClick={ingestFile}
                      disabled={busy === "upload"}
                      variant="primary"
                    >
                      {busy === "upload" ? "Encoding…" : "Upload + ingest"}
                    </Btn>
                  </Field>
                </div>
                <div style={{ marginTop: 13 }}>
                  <Field label="Paste text">
                    <textarea
                      value={ingestText}
                      onChange={(event) => setIngestText(event.target.value)}
                      style={{ minHeight: 120 }}
                    />
                  </Field>
                </div>
                <div className="row" style={{ marginTop: 12 }}>
                  <Btn
                    onClick={ingestPaste}
                    disabled={busy === "ingest" || !ingestText.trim()}
                    variant="primary"
                  >
                    {busy === "ingest" ? "Encoding…" : "Ingest pasted text"}
                  </Btn>
                  <Btn onClick={adapt} disabled={busy === "adapt"}>
                    {busy === "adapt" ? "Adapting…" : `Adapt ${adaptRounds} rounds`}
                  </Btn>
                  <input
                    type="number"
                    min={1}
                    max={200}
                    value={adaptRounds}
                    onChange={(event) => setAdaptRounds(Number(event.target.value))}
                    style={{
                      width: 78,
                      padding: "7px 9px",
                      border: "1px solid var(--line)",
                      borderRadius: 4,
                      background: "#070b10",
                      color: "var(--text)",
                      fontFamily: "var(--font-mono)",
                      fontSize: 12,
                    }}
                  />
                </div>
              </Panel>

              <Panel
                title="Ask the session"
                subtitle="retrieval is real; extractive answers are grounded, generated answers are labelled"
              >
                <div className="form-grid">
                  <Field label="Mode">
                    <select
                      value={mode}
                      onChange={(event) =>
                        setMode(event.target.value as "lexical" | "neural" | "hybrid")
                      }
                    >
                      <option value="lexical">lexical (BM25)</option>
                      <option value="neural">neural (fast weights)</option>
                      <option value="hybrid">hybrid</option>
                    </select>
                  </Field>
                  <Field label="Top k">
                    <input
                      type="number"
                      min={1}
                      max={50}
                      value={topK}
                      onChange={(event) => setTopK(Number(event.target.value))}
                    />
                  </Field>
                  <Field label="Generation memory">
                    <label className="check" style={{ minHeight: 36 }}>
                      <input
                        type="checkbox"
                        checked={useMemory}
                        onChange={(event) => setUseMemory(event.target.checked)}
                      />
                      use fast state in generation
                    </label>
                  </Field>
                </div>
                <div style={{ marginTop: 12 }}>
                  <Field label="Question">
                    <input
                      value={question}
                      onChange={(event) => setQuestion(event.target.value)}
                    />
                  </Field>
                </div>
                <div className="row" style={{ marginTop: 12 }}>
                  <Btn
                    variant="primary"
                    onClick={ask}
                    disabled={busy === "query" || !question.trim()}
                  >
                    {busy === "query" ? "Retrieving…" : "Answer with citations"}
                  </Btn>
                  <Btn onClick={runSearch} disabled={busy === "search"}>
                    {busy === "search" ? "Searching…" : "Show retrieval scores"}
                  </Btn>
                </div>

                {answer ? (
                  <div style={{ marginTop: 16 }}>
                    <div
                      className={`answer ${answer.insufficient ? "insufficient" : ""}`}
                    >
                      <div className="label">
                        {answer.insufficient
                          ? "insufficient information"
                          : answer.answer_type.replaceAll("_", " ")}
                      </div>
                      <div className="text">{answer.answer}</div>
                      <div className="row" style={{ marginTop: 10 }}>
                        <Badge tone={answer.grounded ? "mint" : "amber"}>
                          {answer.grounded ? "grounded" : "unverified generation"}
                        </Badge>
                        <Badge>{answer.mode}</Badge>
                        <Badge>confidence {fmtNum(answer.confidence, 4)}</Badge>
                        <Badge>{fmtNum(answer.latency_ms, 1)} ms</Badge>
                        <Badge tone={answer.memory_used_in_generation ? "cyan" : ""}>
                          {answer.memory_used_in_generation
                            ? "memory in generation"
                            : "no memory in generation"}
                        </Badge>
                      </div>
                      <p className="panel-sub" style={{ marginTop: 10 }}>
                        {answer.notes}
                      </p>
                    </div>
                    <div style={{ marginTop: 12 }}>
                      <div className="stat-label" style={{ marginBottom: 8 }}>
                        citations
                      </div>
                      {answer.sources.length ? (
                        answer.sources.map((source, index) => (
                          <div className="source" key={`${source.chunk_id}-${index}`}>
                            <div className="source-head">
                              <span>
                                {source.filename} · page {source.page ?? "—"} ·{" "}
                                {source.chunk_id}
                              </span>
                              <span>score {fmtNum(source.score, 4)}</span>
                            </div>
                            {source.record_key ? (
                              <div className="mono" style={{ marginTop: 6, fontSize: 10, color: "var(--mint)" }}>
                                {source.record_key} → {source.record_value}
                              </div>
                            ) : null}
                            <p>{source.preview}</p>
                          </div>
                        ))
                      ) : (
                        <Empty>No citations.</Empty>
                      )}
                    </div>
                  </div>
                ) : null}

                {search ? (
                  <div style={{ marginTop: 16 }}>
                    <div className="stat-label" style={{ marginBottom: 8 }}>
                      retrieval scores · {search.mode}
                    </div>
                    {search.results.length ? (
                      <div className="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>chunk</th>
                              <th>source</th>
                              <th className="num">score</th>
                              <th className="num">lexical</th>
                              <th className="num">neural</th>
                              <th>record</th>
                            </tr>
                          </thead>
                          <tbody>
                            {search.results.map((row, index) => (
                              <tr key={`${row.chunk_id}-${index}`}>
                                <td className="mono">{row.chunk_id}</td>
                                <td>
                                  {row.filename} · p{row.page ?? "—"}
                                </td>
                                <td className="num mono">{fmtNum(row.score, 4)}</td>
                                <td className="num mono">{fmtNum(row.lexical_score, 4)}</td>
                                <td className="num mono">{fmtNum(row.neural_score, 4)}</td>
                                <td className="mono">
                                  {row.key_text
                                    ? `${row.key_text} → ${row.value_text}`
                                    : "—"}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <Empty>No retrieval rows.</Empty>
                    )}
                  </div>
                ) : null}
              </Panel>

              <Panel
                title="Documents"
                subtitle="delete removes chunks, records and the rebuilt lexical index"
                flush
              >
                {documents.length ? (
                  <div className="table-wrap" style={{ border: "none" }}>
                    <table>
                      <thead>
                        <tr>
                          <th>filename</th>
                          <th>kind</th>
                          <th className="num">pages</th>
                          <th className="num">chunks</th>
                          <th className="num">bytes</th>
                          <th />
                        </tr>
                      </thead>
                      <tbody>
                        {documents.map((doc) => (
                          <tr key={doc.doc_id}>
                            <td className="mono">{doc.filename}</td>
                            <td>{doc.kind}</td>
                            <td className="num">{doc.pages ?? "—"}</td>
                            <td className="num">{doc.chunk_count ?? "—"}</td>
                            <td className="num">{fmtNum(doc.size_bytes)}</td>
                            <td>
                              <Btn
                                size="sm"
                                variant="danger"
                                onClick={() => removeDocument(doc.doc_id)}
                              >
                                Delete
                              </Btn>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <Empty>No documents in this session.</Empty>
                )}
              </Panel>

              <Panel title="Chunks" subtitle="bounded preview of indexed content" flush>
                {chunks.length ? (
                  <div className="table-wrap" style={{ border: "none" }}>
                    <table>
                      <thead>
                        <tr>
                          <th>chunk</th>
                          <th>source</th>
                          <th className="num">tokens</th>
                          <th className="num">records</th>
                          <th>preview</th>
                        </tr>
                      </thead>
                      <tbody>
                        {chunks.slice(0, 12).map((chunk) => (
                          <tr key={chunk.chunk_id}>
                            <td className="mono">{chunk.chunk_id}</td>
                            <td>
                              {chunk.filename} · p{chunk.page ?? "—"}
                            </td>
                            <td className="num">{chunk.token_count ?? "—"}</td>
                            <td className="num">{chunk.records ?? 0}</td>
                            <td style={{ maxWidth: 420 }}>{chunk.preview}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <Empty>No chunks.</Empty>
                )}
              </Panel>
            </>
          ) : (
            <Panel>
              <Empty>Select or create a session to ingest and query.</Empty>
            </Panel>
          )}
        </div>

        <div className="stack">
          <Panel
            title="Session state"
            subtitle={active ? active.session_id : "no session selected"}
            actions={
              active ? (
                <Badge tone="mint">{active.memory_kind || "fast weights"}</Badge>
              ) : null
            }
          >
            {active ? (
              <>
                <KV
                  rows={[
                    ["name", active.name],
                    ["created", fmtAgo(active.created_at)],
                    ["memory dim", String(active.memory_dim ?? "—")],
                    ["fast params", fmtNum(active.fast_parameters)],
                    ["state bytes", fmtBytes(active.memory_state_bytes)],
                    ["chunk bytes", fmtBytes(active.chunk_state_bytes)],
                    ["total bytes", fmtBytes(active.total_state_bytes)],
                    ["capacity", `${active.chunk_count} / ${active.capacity_chunks ?? "—"} chunks`],
                    ["queries", JSON.stringify(active.retrieval_counts || {})],
                  ]}
                />
                <div className="row" style={{ marginTop: 14 }}>
                  <Btn size="sm" onClick={() => sessionAction("save")} disabled={busy === "save"}>
                    Save
                  </Btn>
                  <Btn size="sm" onClick={() => sessionAction("restore")} disabled={busy === "restore"}>
                    Restore
                  </Btn>
                  <Btn size="sm" onClick={() => sessionAction("reset")} disabled={busy === "reset"}>
                    Reset
                  </Btn>
                  <Btn size="sm" variant="danger" onClick={removeSession} disabled={busy === "delete"}>
                    Delete
                  </Btn>
                </div>
              </>
            ) : (
              <Empty>No active session.</Empty>
            )}
          </Panel>

          <Panel title="Last query" subtitle="persisted in the session summary">
            {active?.last_query ? (
              <KV
                rows={[
                  ["question", String(active.last_query.question ?? "—")],
                  ["answer", String(active.last_query.answer ?? "—")],
                  ["mode", String(active.last_query.mode ?? "—")],
                  ["confidence", fmtNum(active.last_query.confidence, 4)],
                  ["grounded", String(active.last_query.grounded ?? "—")],
                  ["latency", `${fmtNum(active.last_query.latency_ms, 1)} ms`],
                ]}
              />
            ) : (
              <Empty>No query recorded yet.</Empty>
            )}
          </Panel>

          <Panel title="Retrieval contract" subtitle="what the console will not fake">
            <p className="panel-sub" style={{ lineHeight: 1.7 }}>
              Lexical retrieval is BM25. Neural retrieval scores the adapted
              fast-weight prediction against stored value vectors. Hybrid fuses both
              and keeps the best record per chunk. If the top score is below the
              configured threshold, the answer is marked{" "}
              <span className="mono">insufficient</span> instead of hallucinating.
            </p>
            <div className="row" style={{ marginTop: 12 }}>
              <Badge tone="mint">lexical</Badge>
              <Badge tone="cyan">neural</Badge>
              <Badge tone="amber">hybrid</Badge>
            </div>
          </Panel>

          <Panel title="Session isolation" subtitle="why this is not a shared cache">
            <p className="panel-sub" style={{ lineHeight: 1.7 }}>
              Every session serializes its own fast parameters and chunk metadata.
              Switching sessions never mutates another session&apos;s state. Slow
              parameters are never written by adaptation, so a memory session cannot
              silently change the base model.
            </p>
          </Panel>
        </div>
      </div>
    </>
  );
}
