// Free-AI-Titan-23M — Cloudflare edition (talk to Titan in the browser).
//
// * Teacher: gpt-oss-120b on Workers AI (env.AI). titan.py calls POST /v1/teach
//   whenever the local PyTorch model can't elaborate an answer.
// * Memory: TITAN.md (bundled from the repo) lives in Workers KV (env.KNOWLEDGE, key "titan.md")
//   when a KV namespace is bound; without one it is kept in memory, so the Worker
//   deploys and works with zero setup.
// * Every lesson is reported automatically: stored as a KV report, returned as a
//   notice, and (optionally) posted to REPORT_WEBHOOK_URL (Discord/Slack style).
// * Images: your Perchance generator (PERCHANCE_URL). Video: only "animate" works
//   online (in the browser); heavy video generators run on your device (titan.py).
// * Also: web search connector, OpenAI-compatible chat, and an MCP endpoint at /mcp.

import UI_HTML from "./ui.js";
import SEED_MD from "./seed.js";

const NAME = "Free-AI-Titan-23M";
const VERSION = "1.0.0";
const MD_KEY = "titan.md";
const DEFAULT_TEACHER = "@cf/openai/gpt-oss-120b";
export const DEFAULT_PERCHANCE_URL = "https://perchance.org/rg4m14ri"; // Titan's image generator
const perchanceUrl = (env) => env.PERCHANCE_URL || DEFAULT_PERCHANCE_URL;
const UA = `${NAME}/${VERSION}`;

export const DEFAULT_MD = `# TITAN.md — Knowledge of Free-AI-Titan-23M

Titan's memory. Entries are added automatically when the cloud teacher (gpt-oss-120b on
Cloudflare Workers AI) teaches Titan something it couldn't answer.

## Knowledge
`;

export const TEACHER_SYSTEM =
  "You are the teacher of Titan-23M, a tiny 23M-parameter AI that learns from your answers. " +
  "Titan could not answer the user's question by itself. Write the answer Titan should learn: " +
  "correct, self-contained, friendly, plain text (no Markdown headings, no tables), at most 80 words. " +
  "If the question needs real-time information you don't have, say so briefly.";

const VIDEO_BACKENDS = [
  { name: "animate", kind: "image-to-video", online: true, where: "browser", available: false,
    description: "Runs in your browser (pick 'animate — in this browser')." },
  ...["cogvideox", "ltx", "wan", "modelscope", "svd"].map((name) => ({
    name, online: false, where: "device only (run titan.py)", available: false,
    description: "Runs on your own device with titan.py, ideally with a GPU.",
  })),
];

// ---------------------------------------------------------------------------
// TITAN.md parsing + search (mirrors titan.py)
// ---------------------------------------------------------------------------
const STOPWORDS = new Set(`a an the is are was were be been am do does did of to in on at for and or but
with by from as it its this that these those i you he she we they me my your our their what
which who whom how why when where can could would should will shall may might please tell
about there here have has had so if then than just also not no yes hey`.split(/\s+/));

// Same recall logic as titan.py: chat-speak, contractions, synonyms, stemming and typo fixing,
// so reworded questions are answered from TITAN.md instead of calling the cloud teacher.
const CHAT_WORDS = {
  u: "you", ya: "you", yu: "you", ur: "your", yr: "your", r: "are", y: "why", im: "i am",
  thx: "thanks", thnx: "thanks", thanx: "thanks", ty: "thanks", tysm: "thanks", pls: "please",
  plz: "please", wat: "what", wut: "what", wht: "what", hii: "hi", hiii: "hi", hiya: "hi",
  hai: "hi", helo: "hello", hullo: "hello", hellooo: "hello", heya: "hey", yo: "hey",
  sup: "what is up", wassup: "what is up", whatsup: "what is up", gm: "good morning",
  gn: "good night", bye: "goodbye", cya: "goodbye", bb: "goodbye", k: "ok", kk: "ok",
  okay: "ok", okey: "ok", pic: "picture", pics: "pictures", vid: "video", vids: "videos",
};
const SYNONYMS = {
  make: "generate", create: "generate", produce: "generate", draw: "generate", render: "generate",
  made: "generate", makes: "generate", making: "generate", created: "generate", creates: "generate",
  creating: "generate", built: "generate", build: "generate",
  picture: "image", photo: "image", drawing: "image", illustration: "image", art: "image",
  clip: "video", movie: "video", film: "video", animation: "video",
  internet: "web", online: "web", net: "web", browse: "search", google: "search", lookup: "search",
  train: "teacher", trains: "teacher", trained: "teacher", trainer: "teacher", taught: "teacher",
  learns: "learn", learning: "learn",
  params: "parameter", weights: "parameter", big: "size", large: "size",
  install: "download", grab: "download", thank: "thanks",
};
const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
export function normalize(text) {
  let t = String(text).toLowerCase().replace(/\u2019/g, "'");
  t = t.replace(/\b(what|who|how|where|when|that|it|there|here)'?s\b/g, "$1 is")
    .replace(/n't\b/g, " not").replace(/'re\b/g, " are").replace(/'m\b/g, " am")
    .replace(/'ll\b/g, " will").replace(/'ve\b/g, " have");
  const out = [];
  for (const w of t.match(/[a-z0-9]+/g) || [])
    for (const x of (has(CHAT_WORDS, w) ? CHAT_WORDS[w] : w).split(" ")) out.push(has(SYNONYMS, x) ? SYNONYMS[x] : x);
  return out;
}
const words = (t) => normalize(t).filter((w) => !STOPWORDS.has(w));
const stem = (w) => {
  if (w.length <= 3) return w;
  if (w.endsWith("ies") && w.length > 4) return w.slice(0, -3) + "y";
  if (/(sses|ches|shes|xes|zes)$/.test(w)) return w.slice(0, -2);
  if (w.endsWith("s") && !/(ss|us|is)$/.test(w)) return w.slice(0, -1);
  if (w.endsWith("ing") && w.length > 5) return w.slice(0, -3);
  if (w.endsWith("ed") && w.length > 4) return w.slice(0, -2);
  return w;
};
const terms = (t) => words(t).map((w) => { const st = stem(w); return has(SYNONYMS, st) ? SYNONYMS[st] : st; });
const norm = (q) => String(q).trim().toLowerCase().replace(/[?!. ]+$/, "");
// key under which two questions count as the same question ("whats ur name" == "What is your name?")
const qkey = (q) => normalize(q).join(" ") || norm(q);

function editDistance(a, b, limit) {
  if (Math.abs(a.length - b.length) > limit) return limit + 1;
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) cur.push(Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] !== b[j - 1] ? 1 : 0)));
    if (Math.min(...cur) > limit) return limit + 1;
    prev = cur;
  }
  return prev[b.length];
}
function fixTypo(w, vocab) {
  if (vocab.has(w) || w.length < 4 || /^\d+$/.test(w)) return w; // short words are too ambiguous ("cat" vs "chat")
  const limit = w.length < 7 ? 1 : 2;
  let best = w, bestD = limit + 1;
  for (const v of vocab) {
    if (Math.abs(v.length - w.length) <= limit && v[0] === w[0]) {
      const d = editDistance(w, v, limit);
      if (d < bestD) { best = v; bestD = d; }
    }
  }
  return best;
}
const trigrams = (text) => {
  const s = `  ${text} `, g = new Map();
  for (let i = 0; i < s.length - 2; i++) g.set(s.slice(i, i + 3), (g.get(s.slice(i, i + 3)) || 0) + 1);
  return g;
};
const gramSim = (a, b) => {
  let dot = 0; a.forEach((v, k) => { dot += v * (b.get(k) || 0); });
  const na = Math.sqrt([...a.values()].reduce((s, v) => s + v * v, 0)), nb = Math.sqrt([...b.values()].reduce((s, v) => s + v * v, 0));
  return na && nb ? dot / (na * nb) : 0;
};

export function parseEntries(md) {
  const out = [];
  const re = /^### Q:[ \t]*(.+?)[ \t]*\n([\s\S]*?)(?=^### Q:|^## |(?![\s\S]))/gm;
  let m, i = 0;
  while ((m = re.exec(md))) {
    const question = m[1].trim();
    let body = m[2];
    const meta = {};
    for (const mm of body.matchAll(/<!--([\s\S]*?)-->/g))
      for (const part of mm[1].split("|")) {
        const k = part.indexOf(":");
        if (k > 0) meta[part.slice(0, k).trim()] = part.slice(k + 1).trim();
      }
    body = body.replace(/<!--[\s\S]*?-->/g, "").trim();
    if (body.startsWith("A:")) body = body.slice(2).trim();
    if (question && body) out.push({ id: ++i, question, answer: body, meta });
  }
  return out;
}

export function searchEntries(entries, query, k = 3) {
  const n = Math.max(entries.length, 1);
  const docs = entries.map((e) => terms(e.question));
  const ans = entries.map((e) => terms(e.answer));
  const df = new Map();
  docs.forEach((d, i) => new Set([...d, ...ans[i]]).forEach((w) => df.set(w, (df.get(w) || 0) + 1)));
  const idf = (w) => Math.log(1 + n / (df.get(w) || 1));
  const vec = (toks) => {
    const v = new Map();
    toks.forEach((t) => v.set(t, (v.get(t) || 0) + idf(t)));
    const len = Math.sqrt([...v.values()].reduce((s, x) => s + x * x, 0)) || 1;
    v.forEach((x, key) => v.set(key, x / len));
    return v;
  };
  const cos = (a, b) => { let s = 0; a.forEach((x, key) => { s += x * (b.get(key) || 0); }); return s; };
  const forms = entries.map((e) => normalize(e.question).join(" "));
  const form = normalize(query).join(" ");
  const vocab = new Set(df.keys());
  const q = terms(query).map((w) => fixTypo(w, vocab));
  let scored;
  if (q.length) {
    const qv = vec(q);
    scored = entries.map((e, i) => [forms[i] === form ? 1 : 0.8 * cos(qv, vec(docs[i])) + 0.2 * cos(qv, vec(ans[i])), e]);
  } else { // only small words ("who are you"): compare whole normalised sentences
    const g = trigrams(form);
    scored = entries.map((e, i) => [forms[i] === form ? 1 : gramSim(g, trigrams(forms[i])) * (docs[i].length ? 0.8 : 1), e]);
  }
  return scored
    .sort((a, b) => b[0] - a[0])
    .slice(0, k)
    .filter((t) => t[0] > 0)
    .map(([sc, e]) => [Math.round(sc * 1e4) / 1e4, e]);
}

export function entryMd(question, answer, source) {
  answer = String(answer).trim().replace(/\n(?=### |## )/g, "\n​").replace(/<!--/g, "&lt;!--").replace(/-->/g, "--&gt;");
  const added = new Date().toISOString().slice(0, 10);
  return `### Q: ${String(question).split(/\s+/).join(" ").slice(0, 300)}\nA: ${answer}\n<!-- source: ${source} | added: ${added} -->\n`;
}

export function upsertEntry(md, question, answer, source) {
  const entries = parseEntries(md);
  const old = entries.find((e) => norm(e.question) === norm(question)) || entries.find((e) => qkey(e.question) === qkey(question));
  const block = entryMd(old ? old.question : question, answer, source);
  if (old) {
    const esc = old.question.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`^### Q:[ \\t]*${esc}[ \\t]*\\n[\\s\\S]*?(?=^### Q:|^## |(?![\\s\\S]))`, "m");
    return md.replace(re, () => block + "\n");
  }
  return md.replace(/\n*$/, "") + "\n\n" + block;
}

// ---------------------------------------------------------------------------
// Connectors
// ---------------------------------------------------------------------------
export function extractText(res) {
  if (!res) return "";
  if (typeof res === "string") return res.trim();
  if (res.result && typeof res.result === "object") res = res.result;
  if (typeof res.response === "string") return res.response.trim();
  if (res.output_text) return String(res.output_text).trim();
  for (const ch of res.choices || []) {
    if (ch.message && ch.message.content) return String(ch.message.content).trim();
    if (ch.text) return String(ch.text).trim();
  }
  const texts = [];
  for (const item of res.output || [])
    if (item.type === "message") for (const c of item.content || []) if (c.text) texts.push(c.text);
  return texts.join("\n").trim();
}

async function askTeacher(env, question, contextMd = "") {
  const model = env.TEACHER_MODEL || DEFAULT_TEACHER;
  if (!env.AI) throw new Error("the Workers AI binding (AI) is missing, so the teacher can't be reached");
  let user = `Question: ${question}`;
  if (contextMd) user += `\n\nWhat Titan already knows (may be unrelated):\n${contextMd}`;
  const res = await env.AI.run(model, {
    messages: [{ role: "system", content: TEACHER_SYSTEM }, { role: "user", content: user }],
  });
  const answer = extractText(res);
  if (!answer) throw new Error(`empty answer from ${model}`);
  return { answer, model };
}

export async function webSearch(query, limit = 3) {
  const results = [];
  try {
    const r = await fetch("https://api.duckduckgo.com/?" + new URLSearchParams({ q: query, format: "json", no_html: "1", skip_disambig: "1" }),
      { headers: { "User-Agent": UA } });
    const d = await r.json();
    const abs = d.AbstractText || d.Answer;
    if (abs) results.push({ title: d.Heading || query, url: d.AbstractURL || "", snippet: abs, source: "duckduckgo" });
  } catch (e) { /* connector best effort */ }
  try {
    const r = await fetch("https://en.wikipedia.org/w/api.php?" + new URLSearchParams({
      action: "query", generator: "search", gsrsearch: query, gsrlimit: String(limit), prop: "extracts|info",
      exintro: "1", explaintext: "1", exsentences: "3", inprop: "url", format: "json" }), { headers: { "User-Agent": UA } });
    const d = await r.json();
    const pages = Object.values((d.query || {}).pages || {}).sort((a, b) => (a.index || 0) - (b.index || 0));
    for (const p of pages) results.push({ title: p.title, url: p.fullurl || "", snippet: (p.extract || "").trim(), source: "wikipedia" });
  } catch (e) { /* connector best effort */ }
  return { query, results: results.slice(0, limit + 1) };
}

// "What is PyTorch?" -> "PyTorch": simple look-up questions about one short subject.
const LOOKUP_RE = /^\s*(?:(?:what|who|where)\s*(?:is|are|was|were|'s)\s+|define\s+|tell me about\s+|explain\s+|describe\s+|what does\s+)(?:an?\s+|the\s+)?(.+?)(?:\s+mean)?\s*[?.!]*\s*$/i;
export function lookupSubject(question) {
  const m = LOOKUP_RE.exec(String(question));
  if (!m) return null;
  const subj = m[1].trim().replace(/^["']+|["']+$/g, "");
  return subj && subj.split(/\s+/).length <= 5 ? subj : null;
}

/** Answer from Wikipedia when the article is really about the subject: free, no cloud AI call. */
export async function wikipediaAnswer(question, search = webSearch) {
  const subj = lookupSubject(question);
  if (!subj) return null;
  const want = new Set(terms(subj));
  if (!want.size) return null;
  for (const r of (await search(subj, 3)).results || []) {
    const snippet = (r.snippet || "").trim(), got = new Set(terms(r.title || ""));
    if (!snippet || snippet.includes("may refer to") || ![...want].every((w) => got.has(w)) || got.size > want.size + 2) continue;
    const text = snippet.split(/(?<=[.!?])\s+/).slice(0, 3).join(" ").slice(0, 600).trim();
    const src = (r.source || "web").replace(/^./, (c) => c.toUpperCase());
    return { answer: `${text} (Source: ${src}${r.url ? ", " + r.url : ""})`, source: r.source || "web", title: r.title };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Knowledge + learning
// ---------------------------------------------------------------------------
// KV when bound, otherwise an in-memory store (lasts as long as the Worker instance).
const memory = new Map();
const memStore = {
  get: async (k) => (memory.has(k) ? memory.get(k) : null),
  put: async (k, v) => { memory.set(k, v); },
  list: async ({ prefix = "" } = {}) => ({ keys: [...memory.keys()].filter((k) => k.startsWith(prefix)).sort().map((name) => ({ name })) }),
};
const kv = (env) => env.KNOWLEDGE || memStore;

async function getMd(env) {
  const stored = await kv(env).get(MD_KEY);
  if (!stored) return SEED_MD || DEFAULT_MD;
  // A TITAN.md saved by an older version keeps working, and also gets entries added to the repo since.
  const have = new Set(parseEntries(stored).map((e) => qkey(e.question)));
  let md = stored;
  for (const e of parseEntries(SEED_MD || "")) if (!have.has(qkey(e.question))) md = upsertEntry(md, e.question, e.answer, e.meta.source || "seed");
  return md;
}

async function report(env, ctx, rec) {
  const id = `${new Date().toISOString()}:${crypto.randomUUID().slice(0, 8)}`;
  await kv(env).put(`report:${id}`, JSON.stringify(rec), { expirationTtl: 60 * 60 * 24 * 90 });
  if (env.REPORT_WEBHOOK_URL && rec.message) {
    ctx.waitUntil(fetch(env.REPORT_WEBHOOK_URL, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: rec.message, text: rec.message }),
    }).catch(() => {}));
  }
  return id;
}

/** The cloud call: ask gpt-oss-120b, write TITAN.md, report it automatically. */
async function learn(env, ctx, { question, context = "", draft = "", confidence = 0, reason = "", client = "browser" }) {
  const md = await getMd(env);
  if (!context) context = searchEntries(parseEntries(md), question, 3).map(([, e]) => entryMd(e.question, e.answer, e.meta.source || "")).join("\n");
  const { answer, model } = await askTeacher(env, question, context);
  const newMd = upsertEntry(md, question, answer, `cloud:${model}`);
  await kv(env).put(MD_KEY, newMd);
  const entries = parseEntries(newMd);
  const entry = entries.find((e) => qkey(e.question) === qkey(question)) || entries[entries.length - 1];
  const conf = typeof confidence === "number" ? confidence.toFixed(2) : String(confidence);
  const message = `🔔 Automatic report: Titan couldn't elaborate a response to "${question.slice(0, 120)}" ` +
    `(confidence ${conf}${reason ? ", " + reason : ""}, from ${client}). Cloud call → ${model.split("/").pop()} answered → ` +
    `TITAN.md updated (entry #${entry.id}).`;
  const report_id = await report(env, ctx, { question, reason, confidence, client, model, entry: entry.id, draft: String(draft).slice(0, 400), answer, message });
  return { answer, model, report_id, entry, message };
}

// ---------------------------------------------------------------------------
// Chat (browser edition): TITAN.md first, teacher when Titan doesn't know
// ---------------------------------------------------------------------------
const HELP = `Commands:
  /image <prompt>              open your Perchance image generator
  /search <query>              web search (DuckDuckGo + Wikipedia)
  /teach <question> | <answer> teach Titan yourself
  /learn <question>            force a lesson from gpt-oss-120b
  /knowledge                   how much Titan knows
Video: use the Video tab ("animate" runs in your browser; the heavy generators run on your device with titan.py).`;

async function chat(env, ctx, message) {
  const msg = String(message || "").trim();
  if (!msg) return { reply: "Say something!", source: "titan", notices: [] };
  if (msg.startsWith("/")) {
    const sp = msg.indexOf(" ");
    const cmd = (sp < 0 ? msg : msg.slice(0, sp)).toLowerCase();
    const arg = sp < 0 ? "" : msg.slice(sp + 1).trim();
    if (cmd === "/help") return { reply: HELP, source: "command", notices: [] };
    if (cmd === "/knowledge") {
      const es = parseEntries(await getMd(env));
      return { reply: `TITAN.md has ${es.length} entries. Latest: ${es.slice(-5).map((e) => e.question).join("; ")}`, source: "command", notices: [] };
    }
    if (cmd === "/search") {
      const res = await webSearch(arg);
      return { reply: res.results.map((r) => `• ${r.title}: ${r.snippet.slice(0, 300)} ${r.url}`).join("\n") || "No results.", source: "web", notices: [], search: res };
    }
    if (cmd === "/image") return { reply: `Open Titan's Perchance generator (${perchanceUrl(env)}) and use this prompt: ${arg}`, source: "perchance", notices: [], image: imageInfo(env, arg) };
    if (cmd === "/teach") {
      if (!arg.includes("|")) return { reply: "Usage: /teach question | answer", source: "command", notices: [] };
      const [q, a] = arg.split("|", 2).map((s) => s.trim());
      const md = upsertEntry(await getMd(env), q, a, "user");
      await kv(env).put(MD_KEY, md);
      return { reply: "Thanks! Saved to TITAN.md.", source: "command", notices: ["🔔 TITAN.md updated by you."] };
    }
    if (cmd === "/learn") {
      const l = await learn(env, ctx, { question: arg, reason: "lesson requested" });
      return { reply: l.answer, source: `cloud:${l.model}`, confidence: 0, notices: [l.message], learned: true };
    }
    return { reply: `Unknown command ${cmd}. Type /help.`, source: "command", notices: [] };
  }
  const entries = parseEntries(await getMd(env));
  const hits = searchEntries(entries, msg, 3);
  const threshold = parseFloat(env.RECALL_THRESHOLD || "0.55");
  if (hits.length && hits[0][0] >= threshold) {
    return { reply: hits[0][1].answer, source: "knowledge", confidence: +hits[0][0].toFixed(4), notices: [], knowledge: hits[0][1].id };
  }
  const conf = hits.length ? hits[0][0] : 0;
  if (String(env.WEB_FIRST || "true") !== "false") { // free Wikipedia look-up before spending a cloud call
    let found = null;
    try { found = await wikipediaAnswer(msg, env.__search || webSearch); } catch (e) { found = null; }
    if (found) {
      const md = upsertEntry(await getMd(env), msg, found.answer, `web:${found.source}`);
      await kv(env).put(MD_KEY, md);
      const entry = parseEntries(md).find((e) => qkey(e.question) === qkey(msg));
      const message = `🔔 Automatic report: Titan didn't know this yet, so it looked it up on ${found.source.replace(/^./, (c) => c.toUpperCase())} (${found.title}) instead of calling the cloud. TITAN.md updated${entry ? ` (entry #${entry.id})` : ""}.`;
      await report(env, ctx, { question: msg, reason: "nothing about it in TITAN.md", confidence: conf, client: "browser", model: `web:${found.source}`, entry: entry && entry.id, answer: found.answer, message });
      return { reply: found.answer, source: `web:${found.source}`, confidence: +conf.toFixed(4), notices: [message], learned: true, entry: entry && entry.id };
    }
  }
  const l = await learn(env, ctx, { question: msg, confidence: conf, reason: "nothing about it in TITAN.md", client: "browser" });
  return { reply: l.answer, source: `cloud:${l.model}`, confidence: +conf.toFixed(4), notices: [l.message], learned: true, entry: l.entry.id };
}

function imageInfo(env, prompt) {
  return { ok: false, mode: "browser", url: perchanceUrl(env), prompt, message: "Titan's Perchance generator makes the image: paste the prompt there." };
}

// ---------------------------------------------------------------------------
// MCP (Streamable HTTP, JSON responses)
// ---------------------------------------------------------------------------
const MCP_TOOLS = [
  { name: "titan_chat", description: "Chat with Titan-23M (answers from TITAN.md, learns from gpt-oss-120b when it can't).",
    inputSchema: { type: "object", properties: { message: { type: "string" } }, required: ["message"] } },
  { name: "titan_teach", description: "Ask gpt-oss-120b to teach Titan an answer and save it to TITAN.md.",
    inputSchema: { type: "object", properties: { question: { type: "string" } }, required: ["question"] } },
  { name: "titan_web_search", description: "Search the web (DuckDuckGo instant answers + Wikipedia).",
    inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } },
  { name: "titan_knowledge", description: "Read or search Titan's TITAN.md knowledge file.",
    inputSchema: { type: "object", properties: { query: { type: "string" } } } },
  { name: "titan_generate_image", description: "Get the link to Titan's Perchance image generator for a prompt.",
    inputSchema: { type: "object", properties: { prompt: { type: "string" } }, required: ["prompt"] } },
];

async function mcpHandle(env, ctx, msg) {
  const { id, method, params = {} } = msg || {};
  if (id === undefined || id === null) return null;
  const ok = (result) => ({ jsonrpc: "2.0", id, result });
  const text = (t) => ok({ content: [{ type: "text", text: t }] });
  try {
    if (method === "initialize")
      return ok({ protocolVersion: params.protocolVersion || "2025-06-18", capabilities: { tools: { listChanged: false } },
        serverInfo: { name: NAME, version: VERSION } });
    if (method === "ping") return ok({});
    if (method === "tools/list") return ok({ tools: MCP_TOOLS });
    if (method === "tools/call") {
      const a = params.arguments || {};
      switch (params.name) {
        case "titan_chat": { const r = await chat(env, ctx, a.message); return text([r.reply, ...(r.notices || [])].join("\n")); }
        case "titan_teach": { const l = await learn(env, ctx, { question: a.question, reason: "MCP lesson", client: "mcp" }); return text(`${l.answer}\n${l.message}`); }
        case "titan_web_search": return text(JSON.stringify(await webSearch(a.query || ""), null, 1));
        case "titan_knowledge": {
          const md = await getMd(env);
          if (!a.query) return text(md);
          return text(searchEntries(parseEntries(md), a.query, 5).map(([, e]) => entryMd(e.question, e.answer, e.meta.source || "")).join("\n") || "Nothing found.");
        }
        case "titan_generate_image": return text(JSON.stringify(imageInfo(env, a.prompt)));
        default: return { jsonrpc: "2.0", id, error: { code: -32602, message: `unknown tool ${params.name}` } };
      }
    }
    return { jsonrpc: "2.0", id, error: { code: -32601, message: `method not found: ${method}` } };
  } catch (e) {
    return ok({ content: [{ type: "text", text: `error: ${e.message}` }], isError: true });
  }
}

// ---------------------------------------------------------------------------
// HTTP routing
// ---------------------------------------------------------------------------
const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, Mcp-Session-Id",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
};
const json = (obj, status = 200, extra = {}) =>
  new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json; charset=utf-8", ...CORS, ...extra } });

function authorized(request, env, path) {
  if (!env.TITAN_API_KEY) return true;
  if (path === "/v1/chat" && String(env.PUBLIC_CHAT || "true") === "true") return true;
  return request.headers.get("Authorization") === `Bearer ${env.TITAN_API_KEY}`;
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });
    try {
      if (request.method === "GET") {
        if (path === "/" || path === "/index.html") return new Response(UI_HTML, { headers: { "Content-Type": "text/html; charset=utf-8" } });
        if (path === "/v1/health") {
          const es = parseEntries(await getMd(env));
          return json({ ok: true, name: NAME, version: VERSION, edition: "Cloudflare", knowledge_entries: es.length,
            storage: env.KNOWLEDGE ? "kv" : "memory", teacher_ready: !!env.AI,
            teacher: (env.TEACHER_MODEL || DEFAULT_TEACHER).split("/").pop(), perchance_url: perchanceUrl(env) });
        }
        if (path === "/v1/knowledge")
          return new Response(await getMd(env), { headers: { "Content-Type": "text/markdown; charset=utf-8", ...CORS } });
        if (path === "/v1/video/backends") return json({ backends: VIDEO_BACKENDS });
        if (path === "/v1/image/status") return json({ backend: "perchance", generator_url: perchanceUrl(env), configured: true, headless: false });
        if (!authorized(request, env, path)) return json({ error: "unauthorized" }, 401);
        if (path === "/v1/search") return json(await webSearch(url.searchParams.get("q") || ""));
        if (path === "/v1/knowledge/entries") {
          const es = parseEntries(await getMd(env));
          const q = url.searchParams.get("q");
          return json({ entries: q ? searchEntries(es, q, 10).map(([, e]) => e) : es });
        }
        if (path === "/v1/reports") {
          const list = await kv(env).list({ prefix: "report:", limit: 1000 });
          const keys = list.keys.slice(-50).reverse();
          const reports = await Promise.all(keys.map(async (k) => ({ id: k.name.slice(7), ...JSON.parse((await kv(env).get(k.name)) || "{}") })));
          return json({ reports });
        }
        if (path === "/mcp") return json({ error: "use POST for MCP JSON-RPC" }, 405, { Allow: "POST" });
        return json({ error: "not found" }, 404);
      }
      if (request.method !== "POST") return json({ error: "method not allowed" }, 405);
      if (!authorized(request, env, path)) return json({ error: "unauthorized" }, 401);
      const body = await request.json().catch(() => ({}));
      switch (path) {
        case "/v1/teach": {
          const question = String(body.question || "").trim();
          if (!question) return json({ error: "question is required" }, 400);
          if (body.answer) { // a user-provided answer: just store it
            await kv(env).put(MD_KEY, upsertEntry(await getMd(env), question, String(body.answer), body.source || "user"));
            return json({ ok: true, message: "Saved to TITAN.md." });
          }
          const l = await learn(env, ctx, { ...body, question, client: body.client || "api" });
          return json({ answer: l.answer, model: l.model, report_id: l.report_id, entry: l.entry, message: l.message });
        }
        case "/v1/chat":
          return json(await chat(env, ctx, body.message));
        case "/v1/chat/completions": {
          const msgs = body.messages || [];
          const last = [...msgs].reverse().find((m) => m.role === "user") || { content: "" };
          const content = Array.isArray(last.content) ? last.content.map((c) => c.text || "").join(" ") : String(last.content || "");
          const r = await chat(env, ctx, content);
          return json({ id: "chatcmpl-" + crypto.randomUUID().slice(0, 12), object: "chat.completion", created: Math.floor(Date.now() / 1000),
            model: "titan-23m", choices: [{ index: 0, finish_reason: "stop", message: { role: "assistant", content: r.reply } }], titan: r });
        }
        case "/v1/learn": {
          const l = await learn(env, ctx, { question: String(body.question || ""), reason: "lesson requested", client: "api" });
          return json({ reply: l.answer, source: `cloud:${l.model}`, notices: [l.message], learned: true });
        }
        case "/v1/knowledge": { // merge a TITAN.md pushed by titan.py
          let md = await getMd(env);
          const have = new Set(parseEntries(md).map((e) => qkey(e.question)));
          let added = 0;
          for (const e of parseEntries(String(body.markdown || ""))) {
            if (have.has(qkey(e.question))) continue;
            md = upsertEntry(md, e.question, e.answer, e.meta.source || "sync");
            have.add(qkey(e.question));
            added++;
          }
          if (added) await kv(env).put(MD_KEY, md);
          return json({ ok: true, added, entries: parseEntries(md).length });
        }
        case "/v1/image":
          return json(imageInfo(env, String(body.prompt || "")));
        case "/v1/video":
          return json({ error: "Video generators run on your device: use the browser 'animate' option or run titan.py serve." }, 501);
        case "/mcp": {
          const out = Array.isArray(body)
            ? (await Promise.all(body.map((m) => mcpHandle(env, ctx, m)))).filter(Boolean)
            : await mcpHandle(env, ctx, body);
          if (!out || (Array.isArray(out) && !out.length)) return new Response(null, { status: 202, headers: CORS });
          return json(out, 200, { "Mcp-Session-Id": request.headers.get("Mcp-Session-Id") || crypto.randomUUID() });
        }
        default:
          return json({ error: "not found" }, 404);
      }
    } catch (e) {
      return json({ error: e.message || String(e) }, 500);
    }
  },
};
