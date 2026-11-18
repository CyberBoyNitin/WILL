import { chromium } from "playwright-core";

const CONSOLE_URL = process.env.CONSOLE_URL || "http://127.0.0.1:3000";
const API_URL = process.env.API_URL || "http://127.0.0.1:8000";
const CHROME =
  process.env.CHROME_PATH ||
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
const errors = [];
const results = [];

page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
page.on("console", (message) => {
  if (message.type() !== "error") return;
  const text = message.text();
  if (text.includes("favicon") || text.includes("Download the React DevTools")) return;
  errors.push(`console: ${text}`);
});
page.on("response", (response) => {
  if (response.status() === 404) console.log(`HTTP 404 ${response.url()}`);
});

async function check(name, fn) {
  try {
    await fn();
    results.push({ name, ok: true });
    console.log(`PASS ${name}`);
  } catch (error) {
    results.push({ name, ok: false, error: String(error) });
    console.error(`FAIL ${name}: ${error}`);
  }
}

async function open(path) {
  await page.goto(`${CONSOLE_URL}${path}`, { waitUntil: "domcontentloaded" });
  await page.waitForLoadState("networkidle", { timeout: 30000 }).catch(() => {});
  await page.waitForTimeout(900);
}

const sessionName = `ui-smoke-${Date.now()}`;
const trainName = `ui-smoke-train-${Date.now()}`;

try {
  await check("overview hydrates with real model data", async () => {
    await open("/");
    await page.getByText("Adaptive computation, measured.").waitFor();
    await page.getByText("Slow parameters").waitFor();
    await page.getByText("API online").waitFor();
    const body = await page.locator("body").innerText();
    if (!body.includes("581,765")) throw new Error("parameter count missing");
    if (!body.includes("add-single-act-final")) throw new Error("checkpoint missing");
  });

  await check("model page shows architecture and parameter mass", async () => {
    await open("/model");
    await page.getByText("One shared block, variable depth.").waitFor();
    await page.getByText("Parameter mass").waitFor();
    const body = await page.locator("body").innerText();
    if (!body.includes("581,765")) throw new Error("parameter count missing");
    if (!body.includes("Recurrent block")) throw new Error("architecture missing");
  });

  await check("training page renders form and job table", async () => {
    await open("/training");
    await page.getByText("Train the gate, not just the weights.").waitFor();
    await page.getByRole("button", { name: "Start training" }).waitFor();
    await page.getByText("Recent training jobs").waitFor();
  });

  await check("training form queues a real one-epoch run", async () => {
    await open("/training");
    const run = page.locator("section.panel", { hasText: "New run" }).first();
    const field = (label) => run.locator(".field", { hasText: label }).first();
    await field("Execution mode").locator("select").selectOption("fixed_depth");
    await field("Epochs").locator("input").fill("1");
    await field("Train examples").locator("input").fill("8");
    const valTest = field("Val / test examples").locator("input");
    await valTest.nth(0).fill("4");
    await valTest.nth(1).fill("4");
    await field("Batch size").locator("input").fill("4");
    await field("Learning rate").locator("input").fill("0.001");
    await field("Warmup steps").locator("input").fill("0");
    await field("Max steps").locator("input").fill("2");
    await field("Min steps").locator("input").fill("1");
    const complexity = field("Complexity min / max").locator("input");
    await complexity.nth(0).fill("1");
    await complexity.nth(1).fill("1");
    await field("Ponder λ").locator("input").fill("0");
    await field("Checkpoint name").locator("input").fill(trainName);
    await run.getByRole("button", { name: "Start training" }).click();
    const active = page.locator("section.panel", { hasText: "Active job" }).first();
    await active.getByText("completed", { exact: true }).first().waitFor({ timeout: 120000 });
    const text = await active.innerText();
    if (!text.includes(trainName)) throw new Error("training result missing checkpoint");
  });

  await check("benchmarks page renders suites and reports", async () => {
    await open("/benchmarks");
    await page.getByText("Measure compute, not vibes.").waitFor();
    await page.getByRole("button", { name: "Start benchmark" }).waitFor();
    const body = await page.locator("body").innerText();
    if (!body.includes("adaptive_compute")) throw new Error("suite missing");
    if (!body.includes("memory_recall")) throw new Error("suite missing");
  });

  await check("memory create, ingest and grounded answer", async () => {
    await open("/memory");
    await page.getByText("A session that can learn, then forget.").waitFor();
    await page.locator('input[placeholder="session name"]').fill(sessionName);
    await page.getByRole("button", { name: "New session" }).click();
    await page.getByText("Created session", { exact: false }).first().waitFor({ timeout: 20000 });
    const ingest = page.locator("section.panel", { hasText: "Ingest" }).first();
    await ingest.locator("textarea").fill(
      "K00=V100. K01=V101. K02=V102. access code alpha_1234.",
    );
    await ingest.getByRole("button", { name: "Ingest pasted text" }).click();
    await page.getByText("Ingested", { exact: false }).first().waitFor({ timeout: 30000 });
    const ask = page.locator("section.panel", { hasText: "Ask the session" }).first();
    const question = ask.locator(".field", { hasText: "Question" }).locator("input");
    await question.fill("what is the value of K00?");
    await ask.getByRole("button", { name: "Answer with citations" }).click();
    await page.locator(".answer .text").waitFor({ timeout: 30000 });
    const answer = await page.locator(".answer .text").innerText();
    if (!/v100/i.test(answer)) {
      throw new Error(`expected V100 answer, got ${JSON.stringify(answer)}`);
    }
    await page.waitForTimeout(700);
    const state = page.locator("section.panel", { hasText: "Session state" }).first();
    await state.getByRole("button", { name: "Delete", exact: true }).click();
    await page.getByText("Session deleted.").waitFor({ timeout: 15000 });
  });

  await check("playground generates and runs forward diagnostics", async () => {
    await open("/playground");
    await page.getByText("Run the model, inspect the depth.").waitFor();
    await page
      .locator("section.panel", { hasText: "Generation" })
      .first()
      .locator("button.btn.primary", { hasText: "Generate" })
      .click();
    await page.locator(".answer .text").waitFor({ timeout: 60000 });
    const output = await page.locator(".answer .text").innerText();
    if (!output.trim() || output.includes("(empty)")) {
      throw new Error("generation returned no text");
    }
    await page.getByRole("button", { name: "Forward diagnostics" }).click();
    await page.getByRole("button", { name: "Run forward" }).click();
    await page.getByText("mean entropy").waitFor({ timeout: 30000 });
    await page.getByRole("button", { name: "Tokenizer" }).click();
    await page.getByRole("button", { name: "Tokenize", exact: true }).click();
    await page.getByText("tokens", { exact: false }).first().waitFor({ timeout: 15000 });
  });

  await check("experiments page shows checkpoint history and artifacts", async () => {
    await open("/experiments");
    await page.getByText("The evidence trail.").waitFor();
    await page.getByText("Training history").waitFor();
    const body = await page.locator("body").innerText();
    if (!body.includes("add-single-act-final")) throw new Error("checkpoint missing");
    if (!body.includes("Artifact browser")) throw new Error("artifact browser missing");
  });

  await check("benchmark form queues a real memory-recall run", async () => {
    await open("/benchmarks");
    const suite = page.locator(".field", { hasText: "Suite" }).first().locator("select");
    await suite.selectOption("memory_recall");
    const records = page
      .locator(".field", { hasText: "Memory records" })
      .first()
      .locator("input");
    await records.fill("4");
    const rounds = page
      .locator(".field", { hasText: "Adapt rounds" })
      .first()
      .locator("input");
    await rounds.fill("1");
    await page.getByRole("button", { name: "Start benchmark" }).click();
    const active = page.locator("section.panel", { hasText: "Active benchmark" }).first();
    await active.getByText("completed", { exact: true }).first().waitFor({ timeout: 120000 });
    const text = await active.innerText();
    if (!text.includes("memory_recall")) throw new Error("suite missing from result");
    const csvButton = page.getByRole("button", { name: "CSV", exact: true });
    await csvButton.waitFor({ timeout: 30000 });
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      csvButton.click(),
    ]);
    if (!download.suggestedFilename().endsWith(".csv")) {
      throw new Error(`unexpected CSV download ${download.suggestedFilename()}`);
    }
  });
} finally {
  if (errors.length) {
    console.error("\nBrowser errors:");
    for (const error of errors) console.error(`- ${error}`);
  }
  try {
    const response = await fetch(`${API_URL}/api/memory/sessions`);
    const payload = await response.json();
    for (const session of payload.sessions || []) {
      if (session.name === sessionName) {
        await fetch(`${API_URL}/api/memory/sessions/${session.session_id}`, {
          method: "DELETE",
        });
      }
    }
  } catch {
    // cleanup is best-effort
  }
  try {
    const response = await fetch(`${API_URL}/api/checkpoints`);
    const payload = await response.json();
    for (const checkpoint of payload.checkpoints || []) {
      if (checkpoint.checkpoint_id.startsWith(trainName)) {
        await fetch(
          `${API_URL}/api/checkpoints/${encodeURIComponent(checkpoint.checkpoint_id)}`,
          { method: "DELETE" },
        );
      }
    }
  } catch {
    // cleanup is best-effort
  }
  await browser.close();
  const failed = results.filter((item) => !item.ok);
  console.log(
    `\n${results.length - failed.length}/${results.length} console checks passed.`,
  );
  if (failed.length || errors.length) process.exitCode = 1;
}
