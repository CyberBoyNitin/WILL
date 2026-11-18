"use client";

import { useMemo, useState } from "react";

import { DepthStrip, LineChart } from "@/components/Charts";
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
} from "@/components/ui";
import { fmtNum, post } from "@/lib/api";
import { useApi } from "@/lib/hooks";
import type { CheckpointMeta, MemorySessionSummary } from "@/lib/types";

type GenerateResult = {
  checkpoint_id: string;
  task: string | null;
  prompt: string;
  prompt_tokens: number;
  memory_used: boolean;
  result: {
    text: string;
    prompt: string;
    tokens: Array<{ index: number; id: number; text: string; depth: number; kind: string }>;
    prompt_tokens: number;
    generated_tokens: number;
    latency_ms: number;
    tokens_per_second: number | null;
    mean_depth: number;
    max_depth: number;
    token_evaluations: number;
    mode: string;
    memory: any;
    stats: Record<string, any>;
  };
};

type ForwardResult = {
  checkpoint_id: string;
  task: string | null;
  text: string;
  token_count: number;
  logits_shape: number[];
  mean_entropy: number;
  perplexity_proxy: number;
  stats: Record<string, any>;
  tokens: Array<{
    index: number;
    id: number;
    text: string;
    depth: number;
    entropy: number;
    top: Array<{ id: number; text: string; prob: number }>;
  }>;
};

type TokenizeResult = {
  text: string;
  token_count: number;
  ids: number[];
  tokens: string[];
};

const PRESET_FALLBACKS = [
  {
    name: "Single-digit addition",
    prompt: "calc 3 + 6 =",
    task: "arith_add",
  },
  {
    name: "Long-context record lookup",
    prompt:
      "records: r0 amber. r1 basil. access code alpha_1234. question: what is the access code?",
    task: "longctx",
  },
  {
    name: "Multi-hop entity chain",
    prompt:
      "facts: e1 links e2. e2 owns item7. question: who owns item7?",
    task: "multihop",
  },
];

function modeLabel(mode: string) {
  if (mode === "adaptive_hard") return "adaptive_hard · skip halted tokens";
  if (mode === "adaptive_soft") return "adaptive_soft · differentiable ACT";
  return "fixed_depth · baseline";
}

export default function PlaygroundPage() {
  const checkpoints = useApi<{
    checkpoints: CheckpointMeta[];
    default_checkpoint_id: string | null;
  }>("/api/checkpoints", []);
  const sessions = useApi<{ sessions: MemorySessionSummary[] }>(
    "/api/memory/sessions",
    [],
  );
  const presets = useApi<{ presets: any[] }>("/api/playground/presets", []);
  const [tab, setTab] = useState<"generate" | "forward" | "tokenize">("generate");
  const [checkpointId, setCheckpointId] = useState("");
  const [prompt, setPrompt] = useState("calc 3 + 6 =");
  const [mode, setMode] = useState<"fixed_depth" | "adaptive_soft" | "adaptive_hard">(
    "adaptive_hard",
  );
  const [maxSteps, setMaxSteps] = useState("");
  const [haltEpsilon, setHaltEpsilon] = useState("");
  const [temperature, setTemperature] = useState(0);
  const [topK, setTopK] = useState(0);
  const [maxNewTokens, setMaxNewTokens] = useState(24);
  const [appendSep, setAppendSep] = useState(true);
  const [memorySession, setMemorySession] = useState("");
  const [generated, setGenerated] = useState<GenerateResult | null>(null);
  const [forwardText, setForwardText] = useState("calc 3 + 6 =");
  const [forward, setForward] = useState<ForwardResult | null>(null);
  const [tokenText, setTokenText] = useState("WILL learns with fast weights.");
  const [tokenized, setTokenized] = useState<TokenizeResult | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const presetList = presets.data?.presets?.length
    ? presets.data.presets
    : PRESET_FALLBACKS;

  const defaultCheckpoint = checkpoints.data?.default_checkpoint_id || "";

  const outputTokens = useMemo(
    () => generated?.result.tokens || [],
    [generated],
  );

  async function generate() {
    setBusy("generate");
    setError(null);
    try {
      const result = await post<GenerateResult>("/api/playground/generate", {
        prompt,
        checkpoint_id: checkpointId || null,
        mode,
        max_steps: maxSteps ? Number(maxSteps) : null,
        halt_epsilon: haltEpsilon ? Number(haltEpsilon) : null,
        memory_session_id: memorySession || null,
        temperature,
        top_k: topK,
        max_new_tokens: maxNewTokens,
        append_sep: appendSep,
      });
      setGenerated(result);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function runForward() {
    setBusy("forward");
    setError(null);
    try {
      const result = await post<ForwardResult>("/api/playground/forward", {
        text: forwardText,
        checkpoint_id: checkpointId || null,
        mode,
        max_steps: maxSteps ? Number(maxSteps) : null,
        halt_epsilon: haltEpsilon ? Number(haltEpsilon) : null,
        top_k: 5,
      });
      setForward(result);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function runTokenize() {
    setBusy("tokenize");
    setError(null);
    try {
      const result = await post<TokenizeResult>("/api/playground/tokenize", {
        text: tokenText,
      });
      setTokenized(result);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  }

  return (
    <>
      <PageHeader
        kicker="WILL / PLAYGROUND"
        title="Run the model, inspect the depth."
        description="Generation, forward diagnostics and byte tokenisation against the selected checkpoint. Depth and token evaluations are measured at inference; hard early exit actually gathers only the tokens that are still active."
        actions={
          <>
            <Badge tone="mint">live checkpoint</Badge>
            <Badge>{defaultCheckpoint || "no default"}</Badge>
          </>
        }
      />

      {error ? <ErrorBox error={error} /> : null}

      <div className="tabs">
        <button
          className={`tab ${tab === "generate" ? "active" : ""}`}
          onClick={() => setTab("generate")}
        >
          Generate
        </button>
        <button
          className={`tab ${tab === "forward" ? "active" : ""}`}
          onClick={() => setTab("forward")}
        >
          Forward diagnostics
        </button>
        <button
          className={`tab ${tab === "tokenize" ? "active" : ""}`}
          onClick={() => setTab("tokenize")}
        >
          Tokenizer
        </button>
      </div>

      {tab === "generate" ? (
        <div className="split">
          <Panel
            title="Generation"
            subtitle="one sequence at a time; the prompt is truncated to the checkpoint context"
            actions={
              <Btn variant="primary" onClick={generate} disabled={busy === "generate"}>
                {busy === "generate" ? "Running…" : "Generate"}
              </Btn>
            }
          >
            <div className="form-grid">
              <Field label="Checkpoint" hint="blank = current default">
                <select
                  value={checkpointId}
                  onChange={(event) => setCheckpointId(event.target.value)}
                >
                  <option value="">current default</option>
                  {checkpoints.data?.checkpoints?.map((ckpt) => (
                    <option value={ckpt.checkpoint_id} key={ckpt.checkpoint_id}>
                      {ckpt.checkpoint_id}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Execution mode">
                <select
                  value={mode}
                  onChange={(event) =>
                    setMode(
                      event.target.value as
                        | "fixed_depth"
                        | "adaptive_soft"
                        | "adaptive_hard",
                    )
                  }
                >
                  <option value="adaptive_hard">adaptive_hard</option>
                  <option value="adaptive_soft">adaptive_soft</option>
                  <option value="fixed_depth">fixed_depth</option>
                </select>
              </Field>
              <Field label="Memory session" hint="optional fast-weight state">
                <select
                  value={memorySession}
                  onChange={(event) => setMemorySession(event.target.value)}
                >
                  <option value="">no memory</option>
                  {sessions.data?.sessions?.map((session) => (
                    <option value={session.session_id} key={session.session_id}>
                      {session.name} · {session.session_id.slice(0, 8)}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Max steps" hint="blank = checkpoint default">
                <input
                  type="number"
                  min={1}
                  max={64}
                  value={maxSteps}
                  onChange={(event) => setMaxSteps(event.target.value)}
                  placeholder="default"
                />
              </Field>
              <Field label="Halt epsilon" hint="blank = checkpoint default">
                <input
                  type="number"
                  step="0.01"
                  min={0.001}
                  max={0.999}
                  value={haltEpsilon}
                  onChange={(event) => setHaltEpsilon(event.target.value)}
                  placeholder="default"
                />
              </Field>
              <Field label="Max new tokens">
                <input
                  type="number"
                  min={1}
                  max={256}
                  value={maxNewTokens}
                  onChange={(event) => setMaxNewTokens(Number(event.target.value))}
                />
              </Field>
              <Field label="Temperature" hint="0 = argmax">
                <input
                  type="number"
                  step="0.1"
                  min={0}
                  max={4}
                  value={temperature}
                  onChange={(event) => setTemperature(Number(event.target.value))}
                />
              </Field>
              <Field label="Top k" hint="0 = no sampling cutoff">
                <input
                  type="number"
                  min={0}
                  max={260}
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                />
              </Field>
              <Field label="Append separator">
                <label className="check" style={{ minHeight: 36 }}>
                  <input
                    type="checkbox"
                    checked={appendSep}
                    onChange={(event) => setAppendSep(event.target.checked)}
                  />
                  append SEP before generation
                </label>
              </Field>
            </div>

            <div style={{ marginTop: 13 }}>
              <Field label="Prompt">
                <textarea
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  style={{ minHeight: 100 }}
                />
              </Field>
            </div>

            <div className="row" style={{ marginTop: 12 }}>
              {presetList.map((preset: any) => (
                <Btn
                  size="sm"
                  key={preset.name}
                  onClick={() => {
                    setPrompt(preset.prompt);
                    if (preset.checkpoint_id) setCheckpointId(preset.checkpoint_id);
                  }}
                >
                  {preset.name}
                </Btn>
              ))}
            </div>

            <div className="notice" style={{ marginTop: 14 }}>
              {modeLabel(mode)}. Soft ACT trains all steps; hard ACT runs under{" "}
              <span className="mono">no_grad</span> and skips halted positions. On CPU,
              hard skipping can be slower in wall-clock even when token evaluations
              drop.
            </div>
          </Panel>

          <div className="stack">
            <Panel
              title="Output"
              subtitle={generated ? generated.checkpoint_id : "not generated yet"}
              actions={
                generated ? (
                  <Badge tone="mint">{generated.result.mode}</Badge>
                ) : null
              }
            >
              {busy === "generate" ? (
                <Loading label="Running recurrent steps" />
              ) : generated ? (
                <>
                  <div className="answer">
                    <div className="label">generated text</div>
                    <div className="text mono" style={{ whiteSpace: "pre-wrap" }}>
                      {generated.result.text || "(empty)"}
                    </div>
                  </div>
                  <div className="grid grid-2" style={{ marginTop: 12 }}>
                    <KV
                      rows={[
                        ["latency", `${fmtNum(generated.result.latency_ms, 1)} ms`],
                        ["generated tokens", String(generated.result.generated_tokens)],
                        ["mean depth", fmtNum(generated.result.mean_depth, 3)],
                        ["max depth", fmtNum(generated.result.max_depth, 3)],
                        ["token evals", fmtNum(generated.result.token_evaluations)],
                        [
                          "tokens / second",
                          generated.result.tokens_per_second === null
                            ? "—"
                            : fmtNum(generated.result.tokens_per_second, 2),
                        ],
                        ["memory used", String(generated.memory_used)],
                        ["task", generated.task || "—"],
                      ]}
                    />
                    <div>
                      <div className="stat-label" style={{ marginBottom: 6 }}>
                        per-token depth
                      </div>
                      <DepthStrip
                        depths={outputTokens.map((token) => token.depth)}
                        tokens={outputTokens.map((token) => token.text)}
                      />
                      <div className="panel-sub" style={{ marginTop: 4 }}>
                        prompt {generated.result.prompt_tokens} tokens · generated{" "}
                        {generated.result.generated_tokens}
                      </div>
                    </div>
                  </div>
                  <div style={{ marginTop: 14 }}>
                    <div className="stat-label" style={{ marginBottom: 8 }}>
                      token trace
                    </div>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th className="num">#</th>
                            <th className="num">id</th>
                            <th>token</th>
                            <th>kind</th>
                            <th className="num">depth</th>
                          </tr>
                        </thead>
                        <tbody>
                          {outputTokens.slice(-80).map((token) => (
                            <tr key={token.index}>
                              <td className="num mono">{token.index}</td>
                              <td className="num mono">{token.id}</td>
                              <td className="mono">{JSON.stringify(token.text)}</td>
                              <td>{token.kind}</td>
                              <td className="num mono">{fmtNum(token.depth, 3)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  <details style={{ marginTop: 12 }}>
                    <summary
                      className="mono"
                      style={{ color: "var(--muted)", cursor: "pointer", fontSize: 10 }}
                    >
                      RAW GENERATION STATS
                    </summary>
                    <pre style={{ marginTop: 8, maxHeight: 240 }}>
                      {JSON.stringify(generated.result.stats, null, 2)}
                    </pre>
                  </details>
                </>
              ) : (
                <Empty>Run a generation to see the token trace.</Empty>
              )}
            </Panel>

            <Panel title="Checkpoint presets" subtitle="from the playground endpoint">
              {presets.loading ? (
                <Loading />
              ) : presets.error ? (
                <ErrorBox error={presets.error} onRetry={presets.reload} />
              ) : (
                <div className="stack" style={{ gap: 8 }}>
                  {presetList.map((preset: any) => (
                    <div className="source" key={preset.name}>
                      <div className="source-head">
                        <span>{preset.name}</span>
                        <span>{preset.task}</span>
                      </div>
                      <p style={{ WebkitLineClamp: 2 }}>{preset.prompt}</p>
                      <div className="row" style={{ marginTop: 8 }}>
                        <Badge>{preset.checkpoint_id || "no checkpoint"}</Badge>
                        <Btn
                          size="sm"
                          onClick={() => {
                            setPrompt(preset.prompt);
                            if (preset.checkpoint_id)
                              setCheckpointId(preset.checkpoint_id);
                          }}
                        >
                          Use
                        </Btn>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Panel>
          </div>
        </div>
      ) : null}

      {tab === "forward" ? (
        <div className="split">
          <Panel
            title="Forward pass"
            subtitle="logits, halting depth and next-token entropy for every prompt position"
            actions={
              <Btn variant="primary" onClick={runForward} disabled={busy === "forward"}>
                {busy === "forward" ? "Running…" : "Run forward"}
              </Btn>
            }
          >
            <div className="form-grid">
              <Field label="Checkpoint" hint="blank = current default">
                <select
                  value={checkpointId}
                  onChange={(event) => setCheckpointId(event.target.value)}
                >
                  <option value="">current default</option>
                  {checkpoints.data?.checkpoints?.map((ckpt) => (
                    <option value={ckpt.checkpoint_id} key={ckpt.checkpoint_id}>
                      {ckpt.checkpoint_id}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Execution mode">
                <select
                  value={mode}
                  onChange={(event) =>
                    setMode(
                      event.target.value as
                        | "fixed_depth"
                        | "adaptive_soft"
                        | "adaptive_hard",
                    )
                  }
                >
                  <option value="adaptive_soft">adaptive_soft</option>
                  <option value="adaptive_hard">adaptive_hard</option>
                  <option value="fixed_depth">fixed_depth</option>
                </select>
              </Field>
              <Field label="Max steps">
                <input
                  type="number"
                  min={1}
                  max={64}
                  value={maxSteps}
                  onChange={(event) => setMaxSteps(event.target.value)}
                  placeholder="default"
                />
              </Field>
              <Field label="Halt epsilon">
                <input
                  type="number"
                  step="0.01"
                  min={0.001}
                  max={0.999}
                  value={haltEpsilon}
                  onChange={(event) => setHaltEpsilon(event.target.value)}
                  placeholder="default"
                />
              </Field>
            </div>
            <div style={{ marginTop: 13 }}>
              <Field label="Text">
                <textarea
                  value={forwardText}
                  onChange={(event) => setForwardText(event.target.value)}
                  style={{ minHeight: 100 }}
                />
              </Field>
            </div>
            <div className="notice" style={{ marginTop: 14 }}>
              Forward diagnostics run without fast memory. The token depth is the ACT
              halting distribution&apos;s expected step count for each position.
            </div>
          </Panel>

          <div className="stack">
            <Panel
              title="Diagnostics"
              subtitle={forward ? forward.checkpoint_id : "not run yet"}
              actions={forward ? <Badge tone="cyan">{forward.token_count} tokens</Badge> : null}
            >
              {busy === "forward" ? (
                <Loading label="Running forward pass" />
              ) : forward ? (
                <>
                  <KV
                    rows={[
                      ["logits shape", JSON.stringify(forward.logits_shape)],
                      ["mean entropy", fmtNum(forward.mean_entropy, 4)],
                      ["perplexity proxy", fmtNum(forward.perplexity_proxy, 3)],
                      ["task", forward.task || "—"],
                    ]}
                  />
                  <div style={{ marginTop: 14 }}>
                    <div className="stat-label" style={{ marginBottom: 6 }}>
                      depth by position
                    </div>
                    <DepthStrip
                      depths={forward.tokens.map((token) => token.depth)}
                      tokens={forward.tokens.map((token) => token.text)}
                    />
                  </div>
                  <div style={{ marginTop: 10 }}>
                    <LineChart
                      series={[
                        {
                          label: "entropy",
                          values: forward.tokens.map((token) => token.entropy),
                          color: "#64d8ff",
                        },
                        {
                          label: "depth",
                          values: forward.tokens.map((token) => token.depth),
                          color: "#7cf5c8",
                        },
                      ]}
                      height={160}
                      xLabels={forward.tokens.map((token) => String(token.index))}
                    />
                  </div>
                  <details style={{ marginTop: 12 }}>
                    <summary
                      className="mono"
                      style={{ color: "var(--muted)", cursor: "pointer", fontSize: 10 }}
                    >
                      RAW FORWARD STATS
                    </summary>
                    <pre style={{ marginTop: 8, maxHeight: 240 }}>
                      {JSON.stringify(forward.stats, null, 2)}
                    </pre>
                  </details>
                </>
              ) : (
                <Empty>Run a forward pass to inspect halting depth.</Empty>
              )}
            </Panel>

            <Panel title="Per-token predictions" subtitle="top 5 next-token distribution" flush>
              {forward?.tokens?.length ? (
                <div className="table-wrap" style={{ border: "none" }}>
                  <table>
                    <thead>
                      <tr>
                        <th className="num">#</th>
                        <th>token</th>
                        <th className="num">depth</th>
                        <th className="num">entropy</th>
                        <th>top predictions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {forward.tokens.map((token) => (
                        <tr key={token.index}>
                          <td className="num mono">{token.index}</td>
                          <td className="mono">{JSON.stringify(token.text)}</td>
                          <td className="num mono">{fmtNum(token.depth, 3)}</td>
                          <td className="num mono">{fmtNum(token.entropy, 3)}</td>
                          <td className="mono" style={{ fontSize: 10 }}>
                            {token.top
                              .slice(0, 3)
                              .map(
                                (item) =>
                                  `${JSON.stringify(item.text)} ${fmtNum(item.prob, 3)}`,
                              )
                              .join(" · ")}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <Empty>No predictions yet.</Empty>
              )}
            </Panel>
          </div>
        </div>
      ) : null}

      {tab === "tokenize" ? (
        <div className="split">
          <Panel
            title="Byte tokenizer"
            subtitle="no learned vocabulary; every byte is a token"
            actions={
              <Btn variant="primary" onClick={runTokenize} disabled={busy === "tokenize"}>
                {busy === "tokenize" ? "Tokenizing…" : "Tokenize"}
              </Btn>
            }
          >
            <Field label="Text">
              <textarea
                value={tokenText}
                onChange={(event) => setTokenText(event.target.value)}
                style={{ minHeight: 160 }}
              />
            </Field>
            <div className="notice" style={{ marginTop: 14 }}>
              PAD=0, BOS=1, EOS=2, SEP=3, bytes 4..259. The tokenizer is byte-local,
              so any UTF-8 input can be represented without an out-of-vocabulary path.
            </div>
          </Panel>
          <Panel
            title="Tokens"
            subtitle={tokenized ? `${tokenized.token_count} tokens` : "not tokenized yet"}
            flush
          >
            {tokenized ? (
              <>
                <div style={{ padding: 14, borderBottom: "1px solid var(--line)" }}>
                  <div className="row">
                    {tokenized.tokens.map((token, index) => (
                      <span
                        className="pill"
                        key={`${index}-${token}`}
                        title={`id ${tokenized.ids[index]}`}
                      >
                        <span className="mono" style={{ color: "var(--mint)" }}>
                          {tokenized.ids[index]}
                        </span>
                        {JSON.stringify(token)}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="table-wrap" style={{ border: "none" }}>
                  <table>
                    <thead>
                      <tr>
                        <th className="num">#</th>
                        <th className="num">id</th>
                        <th>token</th>
                      </tr>
                    </thead>
                    <tbody>
                      {tokenized.tokens.map((token, index) => (
                        <tr key={`${index}-${tokenized.ids[index]}`}>
                          <td className="num mono">{index}</td>
                          <td className="num mono">{tokenized.ids[index]}</td>
                          <td className="mono">{JSON.stringify(token)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <Empty>Tokenize text to inspect byte ids.</Empty>
            )}
          </Panel>
        </div>
      ) : null}
    </>
  );
}
