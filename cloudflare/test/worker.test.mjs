// Run with: node --test cloudflare/test/
import test from "node:test";
import assert from "node:assert/strict";
import worker, { DEFAULT_PERCHANCE_URL, parseEntries, searchEntries, upsertEntry, extractText, DEFAULT_MD } from "../src/app.js";
import entry from "../src/index.js";

function fakeEnv(extra = {}) {
  const kv = new Map();
  const calls = [];
  return {
    calls, kv,
    AI: { run: async (model, input) => { calls.push({ model, input }); return { choices: [{ message: { content: "Lima is the capital of Peru." } }] }; } },
    KNOWLEDGE: {
      get: async (k) => kv.get(k) ?? null,
      put: async (k, v) => { kv.set(k, v); },
      list: async ({ prefix }) => ({ keys: [...kv.keys()].filter((k) => k.startsWith(prefix)).map((name) => ({ name })) }),
    },
    __search: async () => ({ results: [] }), // no real Wikipedia calls in tests
    ...extra,
  };
}
const ctx = { waitUntil() {} };
const post = (path, body, headers = {}) =>
  new Request("https://titan.example" + path, { method: "POST", headers: { "Content-Type": "application/json", ...headers }, body: JSON.stringify(body) });

test("parse, search and upsert TITAN.md", () => {
  let md = upsertEntry(DEFAULT_MD, "What is the capital of Peru?", "Lima.", "seed");
  md = upsertEntry(md, "Hello", "Hi there!", "seed");
  const es = parseEntries(md);
  assert.equal(es.length, 2);
  assert.equal(es[0].meta.source, "seed");
  assert.equal(searchEntries(es, "capital of peru")[0][1].answer, "Lima.");
  assert.equal(searchEntries(es, "hello")[0][0], 1);
  md = upsertEntry(md, "what is the capital of peru", "Lima, on the Pacific coast.", "cloud:x");
  assert.equal(parseEntries(md).length, 2);
  assert.match(parseEntries(md)[0].answer, /Pacific/);
});

test("extractText handles Workers AI shapes", () => {
  assert.equal(extractText({ response: "a" }), "a");
  assert.equal(extractText({ result: { choices: [{ message: { content: "b" } }] } }), "b");
  assert.equal(extractText({ output: [{ type: "message", content: [{ type: "output_text", text: "c" }] }] }), "c");
});

test("unknown question -> gpt-oss-120b teaches, TITAN.md updated, automatic report", async () => {
  const env = fakeEnv();
  const r = await (await worker.fetch(post("/v1/chat", { message: "What is the capital of Peru?" }), env, ctx)).json();
  assert.equal(r.learned, true);
  assert.equal(r.source, "cloud:@cf/openai/gpt-oss-120b");
  assert.match(r.notices[0], /Automatic report/);
  assert.equal(env.calls[0].model, "@cf/openai/gpt-oss-120b");
  assert.match(env.kv.get("titan.md"), /### Q: What is the capital of Peru\?\nA: Lima is the capital/);
  assert.ok([...env.kv.keys()].some((k) => k.startsWith("report:")));
  // second time it is answered from TITAN.md, no teacher call
  const r2 = await (await worker.fetch(post("/v1/chat", { message: "what is the capital of peru" }), env, ctx)).json();
  assert.equal(r2.source, "knowledge");
  assert.equal(env.calls.length, 1);
});

test("/v1/teach requires the key when set, and returns the answer for titan.py", async () => {
  const env = fakeEnv({ TITAN_API_KEY: "s3cret" });
  assert.equal((await worker.fetch(post("/v1/teach", { question: "Why is the sky blue?" }), env, ctx)).status, 401);
  const r = await (await worker.fetch(post("/v1/teach", { question: "Why is the sky blue?", confidence: 0.1, reason: "low confidence", client: "titan.py" },
    { Authorization: "Bearer s3cret" }), env, ctx)).json();
  assert.equal(r.answer, "Lima is the capital of Peru.");
  assert.ok(r.report_id);
  assert.match(r.message, /titan\.py/);
});

test("knowledge merge, image, video, MCP and UI routes", async () => {
  const env = fakeEnv();
  const m = await (await worker.fetch(post("/v1/knowledge", { markdown: upsertEntry(DEFAULT_MD, "What is my test phrase?", "Hi from the merge test!", "seed") }), env, ctx)).json();
  assert.equal(m.added, 1);
  const img = await (await worker.fetch(post("/v1/image", { prompt: "a cat" }), env, ctx)).json();
  assert.equal(img.url, "https://perchance.org/rg4m14ri");
  assert.equal(DEFAULT_PERCHANCE_URL, img.url);
  assert.equal((await worker.fetch(post("/v1/video", { prompt: "x" }), env, ctx)).status, 501);
  const tools = await (await worker.fetch(post("/mcp", { jsonrpc: "2.0", id: 1, method: "tools/list" }), env, ctx)).json();
  assert.ok(tools.result.tools.some((t) => t.name === "titan_chat"));
  const call = await (await worker.fetch(post("/mcp", { jsonrpc: "2.0", id: 2, method: "tools/call", params: { name: "titan_chat", arguments: { message: "What is my test phrase?" } } }), env, ctx)).json();
  assert.match(call.result.content[0].text, /merge test/);
  const ui = await worker.fetch(new Request("https://titan.example/"), env, ctx);
  assert.match(await ui.text(), /Titan-23M/);
  const h = await (await worker.fetch(new Request("https://titan.example/v1/health"), env, ctx)).json();
  assert.equal(h.teacher, "gpt-oss-120b");
});

test("zero-setup deploy: no KV binding, bundled TITAN.md answers, UI served", async () => {
  const env = fakeEnv();
  delete env.KNOWLEDGE;
  const ui = await worker.fetch(new Request("https://titan.example/"), env, ctx);
  assert.equal(ui.status, 200);
  assert.match(await ui.text(), /Start chatting/);
  const h = await (await worker.fetch(new Request("https://titan.example/v1/health"), env, ctx)).json();
  assert.equal(h.storage, "memory");
  assert.ok(h.knowledge_entries >= 20);
  const r = await (await worker.fetch(post("/v1/chat", { message: "Who is your teacher?" }), env, ctx)).json();
  assert.equal(r.source, "knowledge");
  assert.match(r.reply, /gpt-oss-120b/);
  const learned = await (await worker.fetch(post("/v1/chat", { message: "What is the capital of Peru?" }), env, ctx)).json();
  assert.equal(learned.learned, true);
  const again = await (await worker.fetch(post("/v1/chat", { message: "What is the capital of Peru?" }), env, ctx)).json();
  assert.equal(again.source, "knowledge");
});

test("missing AI binding gives a clear error instead of crashing", async () => {
  const env = fakeEnv();
  delete env.AI;
  const res = await worker.fetch(post("/v1/chat", { message: "Explain quantum entanglement" }), env, ctx);
  assert.equal(res.status, 500);
  assert.match((await res.json()).error, /Workers AI binding/);
});

test("entry module exports only the Worker (the runtime rejects other named exports)", async () => {
  const mod = await import("../src/index.js");
  assert.deepEqual(Object.keys(mod), ["default"]);
  assert.equal(typeof entry.fetch, "function");
});

test("reworded questions are answered from TITAN.md without any cloud call", async () => {
  const env = fakeEnv();
  for (const q of ["whats ur name", "who r u", "how do i make a video", "can you create pictures?", "thx", "what is pytorh"]) {
    const r = await (await worker.fetch(post("/v1/chat", { message: q }), env, ctx)).json();
    assert.equal(r.source, "knowledge", q);
  }
  assert.equal(env.calls.length, 0);
});

test("look-up questions use Wikipedia before the cloud teacher", async () => {
  const env = fakeEnv({ __search: async (q) => ({ results: [{ title: "Alan Turing", url: "https://en.wikipedia.org/wiki/Alan_Turing",
    snippet: "Alan Turing was an English mathematician. He was a pioneer of computer science.", source: "wikipedia" }] }) });
  const r = await (await worker.fetch(post("/v1/chat", { message: "Who was Alan Turing?" }), env, ctx)).json();
  assert.equal(r.source, "web:wikipedia");
  assert.match(r.notices[0], /instead of calling the cloud/);
  assert.equal(env.calls.length, 0);
  assert.match(env.kv.get("titan.md"), /### Q: Who was Alan Turing\?/);
  // a subject the article isn't about still goes to the teacher
  const env2 = fakeEnv({ __search: async () => ({ results: [{ title: "Peru", snippet: "Peru is a country.", source: "wikipedia" }] }) });
  const r2 = await (await worker.fetch(post("/v1/chat", { message: "What is the capital of Peru?" }), env2, ctx)).json();
  assert.equal(r2.source, "cloud:@cf/openai/gpt-oss-120b");
});
