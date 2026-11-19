#!/usr/bin/env python3
"""
Free-AI-Titan-23M — single-file edition.

Everything in one file:
  * Titan, a byte-level transformer written in PyTorch: 23M, 125M, 350M or 1.3B parameters
  * TITAN.md, its Markdown knowledge file (retrieval + training data)
  * Cloud learning: when Titan can't elaborate an answer it calls its teacher,
    gpt-oss-120b on Cloudflare Workers AI, posts an automatic report message,
    appends the answer to TITAN.md and fine-tunes itself in the background
  * Image generator: your Perchance (perchance.org) generator, driven headlessly
  * Video generators that run on your device (CogVideoX, LTX-Video, Wan 2.1,
    ModelScope, Stable Video Diffusion) + "animate", which also works online
  * Web-search connector (DuckDuckGo instant answers + Wikipedia)
  * HTTP API (+ OpenAI-compatible /v1/chat/completions) and a browser chat UI
  * MCP server (stdio and HTTP) and a small MCP client

Quick start:
    pip install torch pillow
    python titan.py train          # train on TITAN.md (--size 23m|125m|350m|1.3b, default auto)
    python titan.py download ltx   # download a video generator (or: all, --deps for its packages)
    python titan.py serve          # API + browser UI on http://127.0.0.1:8023
    python titan.py chat           # terminal chat
    python titan.py mcp            # MCP server over stdio

Configuration is by environment variable, see Config.from_env() or README.md.
"""
from __future__ import annotations

import argparse
import base64
import copy
import datetime as _dt
import html
import io
import json
import math
import mimetypes
import os
import queue
import random
import re
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = "1.0.0"
NAME = "Free-AI-Titan-23M"
USER_AGENT = f"{NAME}/{__version__} (+https://github.com/vendemiaraul-cyber/Free-AI-Titan-23M)"
TEACHER_MODEL = "@cf/openai/gpt-oss-120b"
DEFAULT_PERCHANCE_URL = "https://perchance.org/rg4m14ri"  # Titan's image generator on perchance.org


def log(*args: Any) -> None:
    print("[titan]", *args, file=sys.stderr, flush=True)


def today() -> str:
    return _dt.date.today().isoformat()


# =============================================================================
# 0. Configuration
# =============================================================================

@dataclass
class ModelConfig:
    vocab_size: int = 258        # 256 bytes + BOS + EOS
    n_layer: int = 7
    n_head: int = 8
    d_model: int = 512
    ctx: int = 1536
    dropout: float = 0.0

    @classmethod
    def preset(cls, size: str) -> "ModelConfig":
        p = MODEL_PRESETS[size]
        return cls(n_layer=p["n_layer"], n_head=p["n_head"], d_model=p["d_model"], ctx=p["ctx"])


# Model sizes, all trained from scratch on TITAN.md. Names match the real parameter counts.
# "train_gb" is the memory full training needs (fp32 weights + AdamW state + gradients, before activations).
MODEL_PRESETS: dict[str, dict] = {
    "23m":  {"n_layer": 7,  "n_head": 8,  "d_model": 512,  "ctx": 1536, "params": "23.0M",
             "train_gb": 0.4, "hardware": "any CPU"},
    "125m": {"n_layer": 15, "n_head": 13, "d_model": 832,  "ctx": 2048, "params": "126.7M",
             "train_gb": 2.0, "hardware": "a GPU or Apple Silicon (slow on CPU)"},
    "350m": {"n_layer": 28, "n_head": 16, "d_model": 1024, "ctx": 2048, "params": "355.1M",
             "train_gb": 5.7, "hardware": "an NVIDIA GPU with 12 GB+"},
    "1.3b": {"n_layer": 26, "n_head": 16, "d_model": 2048, "ctx": 2048, "params": "1.31B",
             "train_gb": 21.0, "hardware": "an NVIDIA GPU with 40 GB+"},
}


def resolve_size(size: str) -> str:
    """'auto' picks the biggest preset this machine can train comfortably."""
    size = (size or "auto").lower()
    if size != "auto":
        if size not in MODEL_PRESETS:
            raise SystemExit(f"unknown size {size!r}; choose auto or one of {', '.join(MODEL_PRESETS)}")
        return size
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        # cards report a bit less than their label (a "40 GB" A100 is ~39.5 GiB)
        return "1.3b" if vram >= 38 else "350m" if vram >= 11 else "125m"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "125m"
    return "23m"


@dataclass
class Config:
    home: Path = Path(__file__).resolve().parent
    size: str = "23m"              # TITAN_SIZE: auto, 23m, 125m, 350m or 1.3b (see MODEL_PRESETS)
    model: ModelConfig = field(default_factory=ModelConfig)
    # cloud teacher: either your deployed worker, or Cloudflare's REST API directly
    cloud_url: str = ""            # TITAN_CLOUD_URL  e.g. https://free-ai-titan-23m.<you>.workers.dev
    cloud_key: str = ""            # TITAN_API_KEY    bearer key of your worker (optional)
    cf_account_id: str = ""        # TITAN_CF_ACCOUNT_ID  (direct mode)
    cf_api_token: str = ""         # TITAN_CF_API_TOKEN   (direct mode)
    teacher_model: str = TEACHER_MODEL
    web_fallback: bool = True      # learn from Wikipedia if the cloud is unreachable
    # perchance image generator
    perchance_url: str = DEFAULT_PERCHANCE_URL  # TITAN_PERCHANCE_URL to use a different generator
    perchance_prompt_selector: str = "#prompt, textarea, input[type=text]"
    perchance_negative_selector: str = "#negative"
    perchance_button_selector: str = "#generateBtn, #generate, button:has-text('Generate'), button"
    perchance_timeout: int = 120
    # MCP servers to connect to (client side)
    mcp_servers: list = field(default_factory=list)
    # answering
    conf_threshold: float = 0.75   # below this confidence -> Titan can't elaborate (see Brain.think)
    high_conf: float = 0.90        # confident enough to answer without a knowledge hit
    min_word_validity: float = 0.97  # share of draft words that must exist in TITAN.md
    sim_threshold: float = 0.30    # knowledge hit counts as relevant above this
    recall_threshold: float = 0.55 # strong hit: answer from TITAN.md if the model fails
    temperature: float = 0.6
    top_k: int = 40
    max_new_tokens: int = 480
    # training
    auto_train: bool = True        # bootstrap-train in the background if no checkpoint exists
    bootstrap_steps: int = 300
    learn_steps: int = 30
    batch_size: int = 8
    lr: float = 3e-4
    learn_lr: float = 1.5e-4
    # API
    local_key: str = ""            # TITAN_LOCAL_KEY: require this bearer key on the local API

    @property
    def md_path(self) -> Path:
        return self.home / "TITAN.md"

    @property
    def ckpt_path(self) -> Path:
        # one checkpoint per size, so switching sizes never overwrites another model
        return self.home / ("titan.pt" if self.size == "23m" else f"titan-{self.size}.pt")

    @property
    def reports_path(self) -> Path:
        return self.home / "titan_reports.jsonl"

    @property
    def out_dir(self) -> Path:
        p = self.home / "outputs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def from_env(cls) -> "Config":
        e = os.environ.get
        c = cls()
        if e("TITAN_HOME"):
            c.home = Path(e("TITAN_HOME")).expanduser().resolve()
            c.home.mkdir(parents=True, exist_ok=True)
        c.size = resolve_size(e("TITAN_SIZE", "auto"))
        c.model = ModelConfig.preset(c.size)
        c.cloud_url = e("TITAN_CLOUD_URL", "").rstrip("/")
        c.cloud_key = e("TITAN_API_KEY", "")
        c.cf_account_id = e("TITAN_CF_ACCOUNT_ID", "")
        c.cf_api_token = e("TITAN_CF_API_TOKEN", "")
        c.teacher_model = e("TITAN_TEACHER_MODEL", TEACHER_MODEL)
        c.web_fallback = e("TITAN_WEB_FALLBACK", "1") != "0"
        c.perchance_url = e("TITAN_PERCHANCE_URL", DEFAULT_PERCHANCE_URL)
        c.perchance_prompt_selector = e("TITAN_PERCHANCE_PROMPT_SELECTOR", c.perchance_prompt_selector)
        c.perchance_negative_selector = e("TITAN_PERCHANCE_NEGATIVE_SELECTOR", c.perchance_negative_selector)
        c.perchance_button_selector = e("TITAN_PERCHANCE_BUTTON_SELECTOR", c.perchance_button_selector)
        c.mcp_servers = [u.strip() for u in e("TITAN_MCP_SERVERS", "").split(",") if u.strip()]
        c.auto_train = e("TITAN_AUTO_TRAIN", "1") != "0"
        c.local_key = e("TITAN_LOCAL_KEY", "")
        for name, cast in (("conf_threshold", float), ("sim_threshold", float), ("learn_steps", int),
                           ("bootstrap_steps", int), ("batch_size", int), ("temperature", float)):
            v = e("TITAN_" + name.upper())
            if v:
                setattr(c, name, cast(v))
        return c


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# 1. TITAN.md knowledge base
# =============================================================================

ENTRY_RE = re.compile(r"^### Q:[ \t]*(.+?)[ \t]*\n(.*?)(?=^### Q:|^## |\Z)", re.S | re.M)
META_RE = re.compile(r"<!--(.*?)-->", re.S)
WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = set("""a an the is are was were be been am do does did of to in on at for and or but
with by from as it its this that these those i you he she we they me my your our their what
which who whom how why when where can could would should will shall may might please tell
about there here have has had so if then than just also not no yes hey""".split())

DEFAULT_MD = """# TITAN.md — Knowledge of Free-AI-Titan-23M

Titan's memory. Entries are added automatically when the cloud teacher (gpt-oss-120b on
Cloudflare Workers AI) teaches Titan something it couldn't answer.

## Knowledge
"""


def words(text: str) -> list[str]:
    return [w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS]


def _stem(w: str) -> str:
    for suf in ("ing", "ies", "es", "ed", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


@dataclass
class Entry:
    id: int
    question: str
    answer: str
    meta: dict

    def to_md(self) -> str:
        meta = " | ".join(f"{k}: {v}" for k, v in self.meta.items())
        return f"### Q: {self.question}\nA: {self.answer}\n<!-- {meta} -->\n"


class KnowledgeBase:
    """Parses, searches and appends to TITAN.md."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        if not self.path.exists():
            self.path.write_text(DEFAULT_MD, encoding="utf-8")
        self.entries: list[Entry] = []
        self.reload()

    # -- parsing ---------------------------------------------------------------
    @staticmethod
    def parse(text: str) -> list[Entry]:
        out = []
        for i, m in enumerate(ENTRY_RE.finditer(text)):
            q, body = m.group(1).strip(), m.group(2)
            meta = {}
            for mm in META_RE.finditer(body):
                for part in mm.group(1).split("|"):
                    if ":" in part:
                        k, v = part.split(":", 1)
                        meta[k.strip()] = v.strip()
            body = META_RE.sub("", body).strip()
            if body.startswith("A:"):
                body = body[2:].strip()
            if q and body:
                out.append(Entry(i + 1, q, body, meta))
        return out

    def reload(self) -> None:
        with self.lock:
            self.text = self.path.read_text(encoding="utf-8")
            self.entries = self.parse(self.text)
            self._index()

    def _index(self) -> None:
        docs = [[_stem(w) for w in words(e.question)] for e in self.entries]
        ans = [[_stem(w) for w in words(e.answer)] for e in self.entries]
        n = max(len(docs), 1)
        df: dict[str, int] = {}
        for d, a in zip(docs, ans):
            for w in set(d) | set(a):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + n / c) for w, c in df.items()}
        self.vecs = [(self._vec(d), self._vec(a)) for d, a in zip(docs, ans)]
        self.vocab = set(WORD_RE.findall(self.text.lower()))

    def word_validity(self, text: str) -> float:
        """Share of the words in `text` that Titan has ever seen in TITAN.md (garbled words aren't)."""
        ws = WORD_RE.findall(text.lower())
        if not ws:
            return 0.0
        with self.lock:
            return sum(w in self.vocab for w in ws) / len(ws)

    def _vec(self, toks: list[str]) -> dict:
        v: dict[str, float] = {}
        for t in toks:
            v[t] = v.get(t, 0.0) + self.idf.get(t, math.log(1 + max(len(self.entries), 1)))
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {k: x / norm for k, x in v.items()}

    @staticmethod
    def _cos(a: dict, b: dict) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(x * b.get(k, 0.0) for k, x in a.items())

    def search(self, query: str, k: int = 3) -> list[tuple[float, Entry]]:
        with self.lock:
            qtok = [_stem(w) for w in words(query)]
            if not qtok:  # e.g. "hi" is not a stopword, but "how are you" is all stopwords
                low = query.strip().lower().rstrip("?!. ")
                exact = [(1.0, e) for e in self.entries if e.question.lower().rstrip("?!. ") == low]
                return exact[:k]
            qv = self._vec(qtok)
            scored = []
            for e, (dv, av) in zip(self.entries, self.vecs):
                s = 0.8 * self._cos(qv, dv) + 0.2 * self._cos(qv, av)
                if e.question.strip().lower().rstrip("?!. ") == query.strip().lower().rstrip("?!. "):
                    s = 1.0
                scored.append((s, e))
            scored.sort(key=lambda t: -t[0])
            return [t for t in scored[:k] if t[0] > 0]

    # -- writing ---------------------------------------------------------------
    def find_exact(self, question: str) -> Optional[Entry]:
        q = question.strip().lower().rstrip("?!. ")
        for e in self.entries:
            if e.question.strip().lower().rstrip("?!. ") == q:
                return e
        return None

    def add(self, question: str, answer: str, source: str) -> Entry:
        question = " ".join(question.split())[:300]
        answer = answer.strip()
        answer = re.sub(r"\n(?=### |## )", "\n​", answer)  # never break the file structure
        answer = answer.replace("<!--", "&lt;!--").replace("-->", "--&gt;")
        with self.lock:
            self.reload()
            old = self.find_exact(question)
            meta = {"source": source, "added": today()}
            if old:  # replace the old answer in place
                pattern = re.compile(r"^### Q:[ \t]*" + re.escape(old.question) + r"[ \t]*\n.*?(?=^### Q:|^## |\Z)",
                                     re.S | re.M)
                new = Entry(old.id, old.question, answer, meta).to_md() + "\n"
                text = pattern.sub(lambda _m: new, self.text, count=1)
            else:
                text = self.text.rstrip("\n") + "\n\n" + Entry(0, question, answer, meta).to_md()
            self.path.write_text(text, encoding="utf-8")
            self.reload()
            return self.find_exact(question)  # type: ignore[return-value]

    def merge_markdown(self, text: str) -> int:
        """Add entries from another TITAN.md that we don't have yet. Returns count added."""
        added = 0
        for e in self.parse(text):
            if not self.find_exact(e.question):
                self.add(e.question, e.answer, e.meta.get("source", "sync"))
                added += 1
        return added


# =============================================================================
# 2. Tokenizer + the Titan transformer (PyTorch)
# =============================================================================

BOS, EOS = 256, 257


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(ids: list[int]) -> str:
    return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


def format_prompt(question: str, knowledge: str = "") -> str:
    knowledge = knowledge.replace("\n", " ")
    return f"### Knowledge: {knowledge[:600]}\n### User: {question[:300]}\n### Titan:"


class Attention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.n_head, self.d = c.n_head, c.d_model
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.proj = nn.Linear(c.d_model, c.d_model)
        self.dropout = c.dropout

    def forward(self, x: torch.Tensor, cache: Optional[dict] = None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // self.n_head
        q, k, v = (t.view(B, T, self.n_head, hd).transpose(1, 2) for t in (q, k, v))
        prefix = 0
        if cache is not None:
            if "k" in cache:
                prefix = cache["k"].size(2)
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        if T > 1 and prefix:  # chunk after a prefix: explicit causal mask with offset
            mask = torch.ones(T, prefix + T, dtype=torch.bool, device=x.device).tril(prefix)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=T > 1, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.attn = Attention(c)
        self.mlp = nn.Sequential(nn.Linear(c.d_model, 4 * c.d_model), nn.GELU(),
                                 nn.Linear(4 * c.d_model, c.d_model), nn.Dropout(c.dropout))

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache)
        return x + self.mlp(self.ln2(x))


class Titan(nn.Module):
    """GPT-style decoder over UTF-8 bytes. Default config = ~23M parameters."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.config = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.ctx, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, caches=None, pos_offset: int = 0):
        B, T = idx.shape
        pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for i, blk in enumerate(self.blocks):
            x = blk(x, caches[i] if caches is not None else None)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: list[int], max_new: int = 400, temperature: float = 0.6,
                 top_k: int = 40, stop_text: str = "\n###") -> tuple[list[int], list[float]]:
        """Sample with a KV cache. Returns (new token ids, log-probs under the untempered model)."""
        self.eval()
        dev = next(self.parameters()).device
        ctx = self.config.ctx
        max_new = max(1, min(max_new, ctx // 2))
        prompt = prompt[-(ctx - max_new):]
        caches = [dict() for _ in self.blocks]
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")
        with amp:
            logits, _ = self(torch.tensor([prompt], device=dev), caches=caches)
        logits = logits[0, -1].float()
        pos = len(prompt)
        out, logps = [], []
        stop = encode(stop_text)
        for _ in range(max_new):
            lp = F.log_softmax(logits, dim=-1)
            if temperature <= 0:
                tok = int(torch.argmax(logits))
            else:
                l2 = logits / temperature
                if top_k:
                    kth = torch.topk(l2, min(top_k, l2.numel())).values[-1]
                    l2 = l2.masked_fill(l2 < kth, float("-inf"))
                tok = int(torch.multinomial(F.softmax(l2, dim=-1), 1))
            logps.append(float(lp[tok]))
            if tok == EOS or tok == BOS:
                break
            out.append(tok)
            if stop and out[-len(stop):] == stop:
                out = out[:-len(stop)]
                break
            if pos >= ctx:
                break
            with amp:
                logits, _ = self(torch.tensor([[tok]], device=dev), caches=caches, pos_offset=pos)
            logits = logits[0, -1].float()
            pos += 1
        return out, logps


# =============================================================================
# 3. Training
# =============================================================================

def make_example(e: Entry, with_knowledge: bool, ctx: int) -> tuple[list[int], list[int]]:
    q = random.choice([e.question, e.question.lower(), e.question.rstrip("?!. ")])
    prompt = [BOS] + encode(format_prompt(q, e.answer if with_knowledge else ""))
    target = encode(" " + e.answer[:600]) + [EOS]
    ids = (prompt + target)[: ctx + 1]
    labels = ([-100] * len(prompt) + target)[: ctx + 1]
    return ids[:-1], labels[1:]


def make_batch(entries: list[Entry], ctx: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    exs = [make_example(e, random.random() < 0.5, ctx) for e in entries]
    T = max(len(x) for x, _ in exs)
    X = torch.zeros(len(exs), T, dtype=torch.long)
    Y = torch.full((len(exs), T), -100, dtype=torch.long)
    for i, (x, y) in enumerate(exs):
        X[i, : len(x)] = torch.tensor(x)
        Y[i, : len(y)] = torch.tensor(y)
    return X.to(device), Y.to(device)


def train_steps(model: Titan, entries: list[Entry], steps: int, lr: float, batch_size: int,
                focus: Optional[list[Entry]] = None, on_step: Optional[Callable] = None) -> list[float]:
    """Train `model` in place. With `focus`, half of every batch is the focus entries (replay for the rest)."""
    if not entries:
        return []
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    model.train()
    losses = []
    for step in range(steps):
        if focus:
            nf = max(1, batch_size // 2)
            batch = [random.choice(focus) for _ in range(nf)] + random.sample(entries, min(len(entries), batch_size - nf))
        else:
            batch = random.sample(entries, min(len(entries), batch_size))
        X, Y = make_batch(batch, model.config.ctx, dev)
        warm = min(1.0, (step + 1) / 10)
        for g in opt.param_groups:
            g["lr"] = lr * warm
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            _, loss = model(X, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if on_step:
            on_step(step, losses[-1])
    model.eval()
    return losses


class BackgroundTrainer:
    """Trains a copy of the live model off-thread, then swaps the weights in."""

    def __init__(self, brain: "Brain"):
        self.brain = brain
        self.q: "queue.Queue[dict]" = queue.Queue()
        self.busy = False
        self.pending_ids: set[str] = set()
        self.history: list[dict] = []
        self.thread = threading.Thread(target=self._loop, daemon=True, name="titan-trainer")
        self.thread.start()

    def submit(self, kind: str, questions: Optional[list[str]] = None, steps: Optional[int] = None) -> bool:
        key = kind + ":" + "|".join(questions or [])
        if key in self.pending_ids:
            return False
        self.pending_ids.add(key)
        self.q.put({"key": key, "kind": kind, "questions": questions or [], "steps": steps})
        return True

    def status(self) -> dict:
        return {"busy": self.busy, "queued": self.q.qsize(), "recent": self.history[-5:]}

    def _loop(self):
        while True:
            job = self.q.get()
            self.busy = True
            t0 = time.time()
            try:
                b = self.brain
                entries = list(b.kb.entries)
                focus = [e for e in entries if e.question in job["questions"]] or None
                if job["kind"] == "bootstrap":
                    steps, lr = job["steps"] or b.cfg.bootstrap_steps, b.cfg.lr
                else:
                    steps, lr = job["steps"] or b.cfg.learn_steps, b.cfg.learn_lr
                if b.model.num_params() <= 400_000_000:
                    with b.model_lock:
                        work = copy.deepcopy(b.model)
                    losses = train_steps(work, entries, steps, lr, b.cfg.batch_size, focus=focus)
                    with b.model_lock:
                        b.model.load_state_dict(work.state_dict())
                        b.trained_steps += steps
                else:  # too big to copy: train the live model (chat waits until this job ends)
                    with b.model_lock:
                        losses = train_steps(b.model, entries, steps, lr, b.cfg.batch_size, focus=focus)
                        b.trained_steps += steps
                b.save()
                rec = {"kind": job["kind"], "questions": job["questions"], "steps": steps,
                       "loss": round(sum(losses[-5:]) / max(1, len(losses[-5:])), 4),
                       "seconds": round(time.time() - t0, 1), "finished": time.strftime("%H:%M:%S")}
                self.history.append(rec)
                log(f"training done: {rec}")
            except Exception:  # never kill the thread
                log("training failed:\n" + traceback.format_exc())
            finally:
                self.pending_ids.discard(job["key"])
                self.busy = False


# =============================================================================
# 4. Connectors: HTTP helper, web search, cloud teacher (gpt-oss-120b)
# =============================================================================

def http_json(url: str, data: Any = None, headers: Optional[dict] = None, timeout: int = 60,
              method: Optional[str] = None) -> Any:
    body = None if data is None else json.dumps(data).encode()
    h = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if body is not None:
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    req = urllib.request.Request(url, data=body, headers=h, method=method or ("POST" if body else "GET"))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw.decode() or "null")


def web_search(query: str, limit: int = 3) -> dict:
    """Free web-search connector: DuckDuckGo instant answers + Wikipedia."""
    results, abstract = [], ""
    try:
        d = http_json("https://api.duckduckgo.com/?" + urllib.parse.urlencode(
            {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}), timeout=15)
        abstract = d.get("AbstractText") or d.get("Answer") or ""
        if abstract:
            results.append({"title": d.get("Heading") or query, "url": d.get("AbstractURL", ""),
                            "snippet": abstract, "source": "duckduckgo"})
    except Exception as ex:
        log("duckduckgo failed:", ex)
    try:
        d = http_json("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": limit,
            "prop": "extracts|info", "exintro": 1, "explaintext": 1, "exsentences": 3,
            "inprop": "url", "format": "json"}), timeout=15)
        pages = sorted((d.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
        for p in pages:
            results.append({"title": p.get("title"), "url": p.get("fullurl", ""),
                            "snippet": (p.get("extract") or "").strip(), "source": "wikipedia"})
    except Exception as ex:
        log("wikipedia failed:", ex)
    return {"query": query, "results": results[: limit + 1]}


class TeacherUnavailable(Exception):
    pass


TEACHER_SYSTEM = (
    "You are the teacher of Titan-23M, a small AI (23M to 1.3B parameters) that learns from your answers. "
    "Titan could not answer the user's question by itself. Write the answer Titan should learn: "
    "correct, self-contained, friendly, plain text (no Markdown headings, no tables), at most 80 words. "
    "If the question needs real-time information you don't have, say so briefly."
)


def extract_llm_text(res: Any) -> str:
    """Pull the answer text out of any Workers AI / OpenAI-shaped response."""
    if isinstance(res, dict) and "result" in res and isinstance(res["result"], (dict, str)):
        res = res["result"]
    if isinstance(res, str):
        return res.strip()
    if not isinstance(res, dict):
        return ""
    if isinstance(res.get("response"), str):
        return res["response"].strip()
    if res.get("output_text"):
        return str(res["output_text"]).strip()
    for ch in res.get("choices") or []:
        msg = ch.get("message") or {}
        if msg.get("content"):
            return str(msg["content"]).strip()
        if ch.get("text"):
            return str(ch["text"]).strip()
    texts = []
    for item in res.get("output") or []:  # Responses API shape
        if item.get("type") == "message":
            for c in item.get("content") or []:
                if c.get("type") in ("output_text", "text") and c.get("text"):
                    texts.append(c["text"])
    return "\n".join(texts).strip()


class CloudTeacher:
    """Asks gpt-oss-120b on Cloudflare for an answer, via your Worker or the REST API."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @property
    def mode(self) -> str:
        if self.cfg.cloud_url:
            return "worker"
        if self.cfg.cf_account_id and self.cfg.cf_api_token:
            return "cloudflare-api"
        return "none"

    def teach(self, question: str, context: list[Entry], draft: str, confidence: float, reason: str) -> dict:
        ctx_md = "\n".join(e.to_md() for e in context[:3])
        if self.mode == "worker":
            h = {"Authorization": f"Bearer {self.cfg.cloud_key}"} if self.cfg.cloud_key else {}
            r = http_json(self.cfg.cloud_url + "/v1/teach", {
                "question": question, "context": ctx_md, "draft": draft[:400],
                "confidence": round(confidence, 4), "reason": reason, "client": f"titan.py/{__version__}",
            }, headers=h, timeout=120)
            if not r or not r.get("answer"):
                raise TeacherUnavailable(f"worker returned no answer: {r}")
            return {"answer": r["answer"], "model": r.get("model", self.cfg.teacher_model),
                    "report_id": r.get("report_id", ""), "via": "worker"}
        if self.mode == "cloudflare-api":
            url = (f"https://api.cloudflare.com/client/v4/accounts/{self.cfg.cf_account_id}"
                   f"/ai/run/{self.cfg.teacher_model}")
            user = f"Question: {question}"
            if ctx_md:
                user += f"\n\nWhat Titan already knows (may be unrelated):\n{ctx_md}"
            r = http_json(url, {"messages": [{"role": "system", "content": TEACHER_SYSTEM},
                                             {"role": "user", "content": user}]},
                          headers={"Authorization": f"Bearer {self.cfg.cf_api_token}"}, timeout=120)
            text = extract_llm_text(r)
            if not text:
                raise TeacherUnavailable(f"empty answer from {self.cfg.teacher_model}")
            return {"answer": text, "model": self.cfg.teacher_model, "report_id": "", "via": "cloudflare-api"}
        raise TeacherUnavailable("no cloud configured (set TITAN_CLOUD_URL, or TITAN_CF_ACCOUNT_ID + TITAN_CF_API_TOKEN)")


# =============================================================================
# 5. Image generator — your Perchance generator
# =============================================================================

def _sniff_ext(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return "png"


_COLLECT_JS = """() => Array.from(document.images)
  .filter(i => i.complete && i.naturalWidth >= 256 && i.naturalHeight >= 256)
  .map(i => i.currentSrc || i.src)"""

_BLOB_JS = """async (src) => { const r = await fetch(src); const b = await r.blob();
  return await new Promise(res => { const f = new FileReader(); f.onload = () => res(f.result); f.readAsDataURL(b); }); }"""


class PerchanceImageGenerator:
    """Drives your perchance.org generator in a headless browser and saves the image it makes.

    Your generator needs a prompt box and a button; by default Titan looks for
    #prompt (or the first textarea / text input), an optional #negative box and
    #generateBtn (or the first button). Override with TITAN_PERCHANCE_*_SELECTOR.
    Without Playwright, Titan returns the generator URL to open in a browser instead.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def status(self) -> dict:
        try:
            import playwright  # noqa: F401
            pw = True
        except ImportError:
            pw = False
        return {"backend": "perchance", "generator_url": self.cfg.perchance_url or None,
                "configured": bool(self.cfg.perchance_url), "headless": pw,
                "note": None if pw else "pip install playwright && playwright install chromium for automatic images"}

    def generate(self, prompt: str, negative: str = "", timeout: Optional[int] = None) -> dict:
        url = self.cfg.perchance_url
        if not url:
            return {"ok": False, "mode": "unconfigured",
                    "message": "Set TITAN_PERCHANCE_URL to your perchance.org generator (e.g. https://perchance.org/rg4m14ri)."}
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return {"ok": False, "mode": "browser", "url": url, "prompt": prompt,
                    "message": "Playwright isn't installed, so open your Perchance generator and paste the prompt there."}
        timeout = timeout or self.cfg.perchance_timeout
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 1000})
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                frame = self._find_frame(page, self.cfg.perchance_prompt_selector, 45)
                before = {s for _, s in self._collect(page)}
                frame.fill(self.cfg.perchance_prompt_selector, prompt)
                if negative:
                    try:
                        if frame.query_selector(self.cfg.perchance_negative_selector):
                            frame.fill(self.cfg.perchance_negative_selector, negative)
                    except Exception:
                        pass
                frame.click(self.cfg.perchance_button_selector)
                deadline = time.time() + timeout
                found = None
                while time.time() < deadline and not found:
                    page.wait_for_timeout(1500)
                    new = [(f, s) for f, s in self._collect(page) if s not in before]
                    found = new[-1] if new else None
                if not found:
                    shot = page.screenshot(full_page=True)
                    path = self._save(shot, "png", "perchance_screenshot")
                    return {"ok": False, "mode": "timeout", "url": url, "file": str(path),
                            "message": f"No new image appeared within {timeout}s; saved a screenshot instead."}
                frame_with, src = found
                data = self._fetch(page, frame_with, src)
                path = self._save(data, _sniff_ext(data), "image")
                return {"ok": True, "mode": "headless", "url": url, "file": str(path), "prompt": prompt}
            finally:
                browser.close()

    @staticmethod
    def _find_frame(page, selector: str, seconds: int):
        deadline = time.time() + seconds
        while time.time() < deadline:
            for f in page.frames:
                try:
                    if f.query_selector(selector):
                        return f
                except Exception:
                    pass
            page.wait_for_timeout(500)
        raise RuntimeError(f"couldn't find the prompt box ({selector}) on the Perchance page")

    @staticmethod
    def _collect(page) -> list:
        out = []
        for f in page.frames:
            try:
                out += [(f, s) for s in f.evaluate(_COLLECT_JS) if s]
            except Exception:
                pass
        return out

    @staticmethod
    def _fetch(page, frame, src: str) -> bytes:
        if src.startswith("data:"):
            return base64.b64decode(src.split(",", 1)[1])
        if src.startswith("blob:"):
            return base64.b64decode(frame.evaluate(_BLOB_JS, src).split(",", 1)[1])
        return page.request.get(src).body()

    def _save(self, data: bytes, ext: str, stem: str) -> Path:
        path = self.cfg.out_dir / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.{ext}"
        path.write_bytes(data)
        return path


# =============================================================================
# 6. Video generators — run on your device (only "animate" also works online)
# =============================================================================

VIDEO_BACKENDS: dict[str, dict] = {
    "animate": {"kind": "image-to-video", "online": True, "gpu": False,
                "description": "Ken Burns zoom/pan over an image (your Perchance image or an upload). "
                               "Runs anywhere, also in the browser edition."},
    "cogvideox": {"download_gb": 13.8, "kind": "text-to-video", "online": False, "gpu": True, "model": "THUDM/CogVideoX-2b",
                  "pipeline": "CogVideoXPipeline", "description": "CogVideoX-2B, 6 s clips, ~12 GB VRAM with offload."},
    "ltx": {"download_gb": 28.4, "kind": "text-to-video", "online": False, "gpu": True, "model": "Lightricks/LTX-Video",
            "pipeline": "LTXPipeline", "description": "LTX-Video, fast text-to-video."},
    "wan": {"download_gb": 28.9, "kind": "text-to-video", "online": False, "gpu": True, "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "pipeline": "WanPipeline", "description": "Wan 2.1 1.3B, ~8 GB VRAM."},
    "modelscope": {"download_gb": 3.7, "kind": "text-to-video", "online": False, "gpu": True, "model": "ali-vilab/text-to-video-ms-1.7b",
                   "pipeline": "DiffusionPipeline", "variant": "fp16", "description": "ModelScope 1.7B, small and old but light."},
    "svd": {"download_gb": 4.5, "kind": "image-to-video", "online": False, "gpu": True,
            "model": "stabilityai/stable-video-diffusion-img2vid-xt", "pipeline": "StableVideoDiffusionPipeline",
            "variant": "fp16",
            "gated": True,
            "description": "Stable Video Diffusion XT: animates an image (accept its license on Hugging Face first)."},
}


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


# What the heavy video generators need (pip names). `python titan.py download --deps` installs them.
VIDEO_PIP = ["diffusers", "transformers", "accelerate", "sentencepiece", "ftfy", "imageio", "imageio-ffmpeg",
             "huggingface_hub"]


def select_video_files(files: list[tuple[str, int]], components: set[str], variant: Optional[str]) -> list[tuple[str, int]]:
    """Pick exactly the files diffusers loads: model_index.json + the pipeline's component folders, one weight
    format per folder (safetensors over .bin; the fp16 variant only when the pipeline loads it). This skips the
    extra single-file checkpoints, example images and duplicate formats many repos also contain."""
    by_dir: dict[str, list[tuple[str, int]]] = {}
    out = [f for f in files if f[0] == "model_index.json"]
    for path, size in files:
        top = path.split("/", 1)[0]
        if "/" in path and top in components:
            by_dir.setdefault(top, []).append((path, size))
    for top, items in by_dir.items():
        weights = [f for f in items if f[0].endswith((".safetensors", ".bin"))]
        others = [f for f in items if f not in weights]
        if any(f[0].endswith(".safetensors") for f in weights):
            weights = [f for f in weights if f[0].endswith(".safetensors")]
        tagged = [f for f in weights if variant and f".{variant}." in f[0]]
        weights = tagged or [f for f in weights if ".fp16." not in f[0]]
        out += others + weights
    return out


class VideoDownloader:
    """Downloads video generator weights from Hugging Face into the normal HF cache, with progress."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.jobs: dict[str, dict] = {}
        self._sizes: dict[str, list] = {}
        self.lock = threading.Lock()

    @property
    def marker(self) -> Path:
        return self.cfg.home / "titan_downloads.json"

    def _completed(self) -> dict:
        try:
            return json.loads(self.marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    # the Hugging Face calls (replaced in tests)
    def _list_files(self, repo: str) -> list[tuple[str, int]]:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True)
        return [(f.rfilename, int(f.size or 0)) for f in info.siblings]

    def _components(self, repo: str) -> set[str]:
        from huggingface_hub import hf_hub_download
        index = json.loads(Path(hf_hub_download(repo, "model_index.json")).read_text(encoding="utf-8"))
        return {k for k, v in index.items() if isinstance(v, list) and v and v[0]}

    def _fetch(self, repo: str, path: str) -> None:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo, path)

    def files(self, name: str) -> list[tuple[str, int]]:
        if name not in self._sizes:
            spec = VIDEO_BACKENDS[name]
            self._sizes[name] = select_video_files(self._list_files(spec["model"]), self._components(spec["model"]),
                                                   spec.get("variant"))
        return self._sizes[name]

    def status(self, name: str) -> dict:
        spec = VIDEO_BACKENDS[name]
        if "model" not in spec:
            return {"state": "built-in", "note": "nothing to download"}
        with self.lock:
            job = dict(self.jobs.get(name) or {})
        if job:
            return job
        done = self._completed().get(name)
        if done:
            return {"state": "downloaded", "bytes_total": done.get("bytes", 0), "finished": done.get("finished")}
        return {"state": "not downloaded", "bytes_total": None}

    def start(self, name: str) -> dict:
        if name not in VIDEO_BACKENDS or "model" not in VIDEO_BACKENDS[name]:
            raise ValueError(f"{name!r} has nothing to download; choose from "
                             + ", ".join(n for n, s in VIDEO_BACKENDS.items() if "model" in s))
        with self.lock:
            if (self.jobs.get(name) or {}).get("state") == "downloading":
                return dict(self.jobs[name])
            self.jobs[name] = {"state": "downloading", "bytes_done": 0, "bytes_total": None,
                               "files_done": 0, "files_total": None, "current": None}
        threading.Thread(target=self._run_safe, args=(name,), daemon=True, name=f"download-{name}").start()
        return self.status(name)

    def _run_safe(self, name: str) -> None:
        try:
            self.download(name)
        except Exception as ex:
            msg = str(ex)
            if "401" in msg or "403" in msg or "gated" in msg.lower():
                msg += " (this model is gated: accept its license on huggingface.co and set HF_TOKEN)"
            with self.lock:
                self.jobs[name] = {**self.jobs.get(name, {}), "state": "error", "error": msg}
            log(f"download of {name} failed: {msg}")

    def download(self, name: str, on_file: Optional[Callable[[str, int, int, int], None]] = None) -> dict:
        """Blocking download (used by the CLI and the background thread)."""
        repo = VIDEO_BACKENDS[name]["model"]
        files = self.files(name)
        total = sum(sz for _, sz in files)
        done_bytes = 0
        with self.lock:
            self.jobs[name] = {"state": "downloading", "bytes_done": 0, "bytes_total": total,
                               "files_done": 0, "files_total": len(files), "current": None}
        for i, (path, size) in enumerate(files):
            with self.lock:
                self.jobs[name]["current"] = path
            self._fetch(repo, path)
            done_bytes += size
            with self.lock:
                self.jobs[name].update(bytes_done=done_bytes, files_done=i + 1)
            if on_file:
                on_file(path, i + 1, len(files), done_bytes)
        completed = self._completed()
        completed[name] = {"repo": repo, "bytes": total, "finished": _dt.datetime.now().isoformat(timespec="seconds")}
        self.marker.write_text(json.dumps(completed, indent=1), encoding="utf-8")
        with self.lock:
            self.jobs.pop(name, None)
        log(f"downloaded {name} ({repo}, {total / 1e9:.1f} GB)")
        return self.status(name)


class VideoGenerator:
    def __init__(self, cfg: Config, images: PerchanceImageGenerator):
        self.cfg, self.images = cfg, images
        self._pipes: dict[str, Any] = {}
        self.lock = threading.Lock()
        self.downloads = VideoDownloader(cfg)

    def backends(self) -> list[dict]:
        cuda = torch.cuda.is_available()
        out = []
        for name, spec in VIDEO_BACKENDS.items():
            deps = ["PIL"] if name == "animate" else ["diffusers", "transformers", "huggingface_hub", "PIL"]
            missing = [d for d in deps if not _has(d)]
            out.append({"name": name, **{k: v for k, v in spec.items() if k != "pipeline"},
                        "available": not missing, "missing": missing,
                        "where": "device + browser" if spec["online"] else "device only",
                        "download": self.downloads.status(name),
                        "pip": None if not missing else "pip install " + " ".join(VIDEO_PIP),
                        "warning": None if (not spec["gpu"] or cuda) else "no CUDA GPU found: will be very slow on CPU"})
        return out

    def generate(self, prompt: str, backend: str = "animate", image: Optional[bytes] = None,
                 frames: Optional[int] = None, fps: int = 12, steps: Optional[int] = None,
                 seed: Optional[int] = None, negative: str = "") -> dict:
        if backend not in VIDEO_BACKENDS:
            raise ValueError(f"unknown video backend {backend!r}; choose from {list(VIDEO_BACKENDS)}")
        seed = random.randint(0, 2**31 - 1) if seed is None else seed
        notes = []
        pil = None
        if image:
            from PIL import Image
            pil = Image.open(io.BytesIO(image)).convert("RGB")
        elif VIDEO_BACKENDS[backend]["kind"] == "image-to-video":
            r = self.images.generate(prompt) if self.cfg.perchance_url else {"ok": False}
            if r.get("ok"):
                from PIL import Image
                pil = Image.open(r["file"]).convert("RGB")
                notes.append(f"first frame generated by Perchance: {Path(r['file']).name}")
            else:
                pil = title_card(prompt)
                notes.append("no image given and Perchance unavailable: animated a title card")
        with self.lock:
            if backend == "animate":
                vid = animate_image(pil, frames or 48)
                fps = fps or 12
            else:
                vid, fps = self._diffusers(backend, prompt, pil, frames, fps, steps, seed, negative)
        path = save_video(vid, fps, self.cfg.out_dir / f"video_{backend}_{time.strftime('%Y%m%d_%H%M%S')}")
        return {"ok": True, "backend": backend, "file": str(path), "frames": len(vid), "fps": fps,
                "seed": seed, "notes": notes}

    def _diffusers(self, name, prompt, pil, frames, fps, steps, seed, negative):
        import diffusers
        spec = VIDEO_BACKENDS[name]
        cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if (cuda and name in ("cogvideox", "ltx", "wan")) else (torch.float16 if cuda else torch.float32)
        pipe = self._pipes.get(name)
        if pipe is None:
            log(f"loading {spec['model']} (first run downloads the weights)…")
            kw: dict[str, Any] = {"torch_dtype": dtype}
            if name == "wan":
                kw["vae"] = diffusers.AutoencoderKLWan.from_pretrained(spec["model"], subfolder="vae",
                                                                       torch_dtype=torch.float32)
            if spec.get("variant"):
                kw["variant"] = spec["variant"]  # the same files `titan.py download` fetches
            pipe = getattr(diffusers, spec["pipeline"]).from_pretrained(spec["model"], **kw)
            if cuda:
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pipe.to("cuda")
            else:
                pipe.to(pick_device() if pick_device().type == "mps" else "cpu")
            for m in ("enable_vae_slicing", "enable_vae_tiling"):
                if hasattr(pipe, m) and name != "svd":
                    try:
                        getattr(pipe, m)()
                    except Exception:
                        pass
            self._pipes = {name: pipe}  # keep only one big model in memory
        g = torch.Generator(device="cpu").manual_seed(seed)
        neg = negative or "blurry, low quality, distorted, watermark, text"
        if name == "svd":
            res = pipe(pil.resize((1024, 576)), num_frames=frames or 25, decode_chunk_size=4, generator=g,
                       num_inference_steps=steps or 25)
            fps = fps or 7
        elif name == "cogvideox":
            res = pipe(prompt=prompt, num_frames=frames or 49, num_inference_steps=steps or 50,
                       guidance_scale=6.0, generator=g)
            fps = fps or 8
        elif name == "ltx":
            n = frames or 97
            n = (n - 1) // 8 * 8 + 1
            res = pipe(prompt=prompt, negative_prompt=neg, width=704, height=480, num_frames=n,
                       num_inference_steps=steps or 40, generator=g)
            fps = fps or 24
        elif name == "wan":
            n = frames or 33
            n = (n - 1) // 4 * 4 + 1
            res = pipe(prompt=prompt, negative_prompt=neg, height=480, width=832, num_frames=n,
                       guidance_scale=5.0, num_inference_steps=steps or 30, generator=g)
            fps = fps or 16
        else:  # modelscope
            res = pipe(prompt, num_frames=frames or 16, num_inference_steps=steps or 25, generator=g)
            fps = fps or 8
        return _to_pil_frames(res.frames[0]), fps


def _to_pil_frames(frames) -> list:
    from PIL import Image
    out = []
    for f in frames:
        if isinstance(f, Image.Image):
            out.append(f.convert("RGB"))
        else:
            import numpy as np
            a = np.asarray(f)
            if a.dtype != np.uint8:
                a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
            out.append(Image.fromarray(a).convert("RGB"))
    return out


def title_card(text: str, size=(768, 432)):
    from PIL import Image, ImageDraw
    w, h = size
    img = Image.new("RGB", size)
    d = ImageDraw.Draw(img)
    for y in range(h):  # vertical gradient
        t = y / h
        d.line([(0, y), (w, y)], fill=(int(20 + 40 * t), int(24 + 30 * t), int(60 + 120 * t)))
    lines, cur = [], ""
    for wd in text.split():
        if len(cur) + len(wd) > 34:
            lines.append(cur)
            cur = wd
        else:
            cur = (cur + " " + wd).strip()
    lines.append(cur)
    y0 = h // 2 - 12 * len(lines[:6])
    for i, ln in enumerate(lines[:6]):
        d.text((48, y0 + 24 * i), ln, fill=(240, 240, 255))
    return img


def animate_image(img, n_frames: int = 48, size=(768, 432)) -> list:
    """Ken Burns effect: slow zoom-in with a gentle pan."""
    from PIL import Image
    W, H = size
    src = img.convert("RGB")
    # cover-fit the source to the output aspect
    scale = max(W / src.width, H / src.height) * 1.0
    src = src.resize((max(W, int(src.width * scale)), max(H, int(src.height * scale))), Image.LANCZOS)
    frames = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        e = t * t * (3 - 2 * t)                        # ease in-out
        z = 1.0 + 0.22 * e
        cw, ch = src.width / z, src.height / z
        cx = src.width / 2 + (src.width - cw) * 0.25 * (e - 0.5)
        cy = src.height / 2 - (src.height - ch) * 0.15 * (e - 0.5)
        box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
        frames.append(src.resize((W, H), Image.LANCZOS, box=box))
    return frames


def save_video(frames: list, fps: int, stem: Path) -> Path:
    """MP4 when imageio/opencv is available (via diffusers or imageio), else animated GIF."""
    try:
        import imageio.v2 as imageio  # needs imageio-ffmpeg for mp4
        import numpy as np
        path = stem.with_suffix(".mp4")
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8) as w:
            for f in frames:
                w.append_data(np.asarray(f))
        return path
    except Exception:
        pass
    path = stem.with_suffix(".gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / max(1, fps)), loop=0)
    return path


# =============================================================================
# 7. MCP client (connect Titan to other MCP servers over Streamable HTTP)
# =============================================================================

class MCPClient:
    def __init__(self, url: str, headers: Optional[dict] = None):
        self.url, self.headers = url, dict(headers or {})
        self.session_id: Optional[str] = None
        self._id = 0
        self.ready = False

    def _post(self, payload: dict) -> Optional[dict]:
        h = {"User-Agent": USER_AGENT, "Content-Type": "application/json",
             "Accept": "application/json, text/event-stream", **self.headers}
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=h, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            sid = r.headers.get("Mcp-Session-Id")
            if sid:
                self.session_id = sid
            ctype = r.headers.get("Content-Type", "")
            raw = r.read().decode()
        if "id" not in payload or not raw.strip():
            return None
        if "text/event-stream" in ctype:
            for line in raw.splitlines():
                if line.startswith("data:"):
                    msg = json.loads(line[5:].strip())
                    if msg.get("id") == payload["id"]:
                        return msg
            raise RuntimeError("no matching response in event stream")
        return json.loads(raw)

    def rpc(self, method: str, params: Optional[dict] = None) -> Any:
        if not self.ready and method != "initialize":
            self.initialize()
        self._id += 1
        msg = self._post({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})
        if msg is None:
            return None
        if "error" in msg:
            raise RuntimeError(f"MCP error: {msg['error']}")
        return msg.get("result")

    def initialize(self) -> dict:
        res = self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                      "clientInfo": {"name": NAME, "version": __version__}})
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.ready = True
        return res

    def list_tools(self) -> list[dict]:
        return (self.rpc("tools/list") or {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self.rpc("tools/call", {"name": name, "arguments": arguments}) or {}


# =============================================================================
# 8. The Brain: answer → confidence check → cloud learning → automatic message
# =============================================================================

HELP = """Commands:
  /image <prompt>            make an image with your Perchance generator
  /video <prompt>            make a video (animate); /video:<backend> <prompt> for cogvideox, ltx, wan, modelscope, svd
  /search <query>            web search (DuckDuckGo + Wikipedia)
  /teach <question> | <answer>   teach Titan yourself
  /learn <question>          force a cloud lesson from gpt-oss-120b
  /tools                     list tools of connected MCP servers
  /tool <name> <json args>   call an MCP tool
  /knowledge                 how much Titan knows
  /status                    model + training status"""


class Brain:
    def __init__(self, cfg: Optional[Config] = None, load: bool = True):
        self.cfg = cfg or Config.from_env()
        self.device = pick_device()
        self.kb = KnowledgeBase(self.cfg.md_path)
        self.model_lock = threading.Lock()
        self.trained_steps = 0
        self.model = Titan(self.cfg.model).to(self.device)
        if load and self.cfg.ckpt_path.exists():
            self.load()
        self.teacher = CloudTeacher(self.cfg)
        self.images = PerchanceImageGenerator(self.cfg)
        self.videos = VideoGenerator(self.cfg, self.images)
        self.mcp = {u: MCPClient(u) for u in self.cfg.mcp_servers}
        self.trainer = BackgroundTrainer(self)
        self.report_lock = threading.Lock()
        if self.trained_steps == 0 and self.cfg.auto_train and self.kb.entries:
            log(f"no checkpoint yet: bootstrap training on {len(self.kb.entries)} TITAN.md entries in the background")
            self.trainer.submit("bootstrap")

    # -- persistence -------------------------------------------------------------
    def save(self) -> None:
        with self.model_lock:
            state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        tmp = self.cfg.ckpt_path.with_suffix(".tmp")
        torch.save({"version": __version__, "config": asdict(self.cfg.model), "state_dict": state,
                    "trained_steps": self.trained_steps}, tmp)
        os.replace(tmp, self.cfg.ckpt_path)

    def load(self) -> None:
        ck = torch.load(self.cfg.ckpt_path, map_location="cpu", weights_only=True)
        mc = ModelConfig(**ck["config"])
        if mc != self.cfg.model:
            log(f"{self.cfg.ckpt_path.name} was trained with a different shape; using the checkpoint's")
            self.cfg.model = mc
            self.model = Titan(mc).to(self.device)
        self.model.load_state_dict(ck["state_dict"])
        self.trained_steps = int(ck.get("trained_steps", 0))
        log(f"loaded {self.cfg.ckpt_path.name} ({self.trained_steps} training steps)")

    def info(self) -> dict:
        return {"name": NAME, "version": __version__, "edition": "single-file (titan.py)",
                "size": self.cfg.size, "parameters": self.model.num_params(), "config": asdict(self.cfg.model),
                "checkpoint": self.cfg.ckpt_path.name,
                "device": str(self.device), "trained_steps": self.trained_steps,
                "knowledge_entries": len(self.kb.entries), "teacher": {"model": self.cfg.teacher_model,
                                                                        "mode": self.teacher.mode},
                "image": self.images.status(), "mcp_servers": list(self.mcp), "training": self.trainer.status()}

    # -- core ----------------------------------------------------------------------
    def think(self, question: str, knowledge: str = "") -> tuple[str, float]:
        """Generate a draft. Confidence = geometric-mean token probability x word validity^3.

        A byte-level model can be "sure" about most bytes of a garbled word, so token probability
        alone is too optimistic; requiring the words to be real (seen in TITAN.md) fixes that.
        """
        prompt = [BOS] + encode(format_prompt(question, knowledge))
        with self.model_lock:
            ids, logps = self.model.generate(prompt, self.cfg.max_new_tokens, self.cfg.temperature, self.cfg.top_k)
        text = decode(ids).strip()
        token_conf = math.exp(sum(logps) / len(logps)) if logps else 0.0
        return text, token_conf * self.kb.word_validity(text) ** 3  # cubed: a few broken words matter

    def report(self, rec: dict) -> None:
        rec = {"time": _dt.datetime.now().isoformat(timespec="seconds"), **rec}
        with self.report_lock, open(self.cfg.reports_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def reports(self, limit: int = 50) -> list[dict]:
        if not self.cfg.reports_path.exists():
            return []
        lines = self.cfg.reports_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(x) for x in lines[-limit:] if x.strip()]

    def learn(self, question: str, hits: list[tuple[float, Entry]], draft: str, conf: float, reason: str) -> dict:
        """The cloud call: ask the teacher, update TITAN.md, train, and write the automatic message."""
        ctx = [e for _, e in hits]
        teacher_err = None
        try:
            lesson = self.teacher.teach(question, ctx, draft, conf, reason)
            source = f"cloud:{lesson['model']}"
            who = f"{lesson['model'].split('/')[-1]} on Cloudflare"
        except Exception as ex:
            teacher_err = str(ex)
            lesson = None
            if self.cfg.web_fallback:
                res = web_search(question, limit=1).get("results", [])
                if res and res[0].get("snippet"):
                    r0 = res[0]
                    lesson = {"answer": f"{r0['snippet']} (source: {r0.get('url') or r0['source']})",
                              "model": r0["source"], "report_id": "", "via": "web"}
                    source, who = f"web:{r0['source']}", f"the web ({r0['source']})"
        if lesson is None:
            self.report({"question": question, "reason": reason, "confidence": round(conf, 4),
                         "status": "pending", "error": teacher_err})
            msg = (f"🔔 Automatic report: Titan couldn't elaborate a response (confidence {conf:.2f}, {reason}) "
                   f"and the cloud teacher is unreachable ({teacher_err}). The question is queued; "
                   f"run `python titan.py sync` when you're online to learn it.")
            return {"reply": "I can't answer that yet, and I couldn't reach my teacher. I've queued the question "
                             "so I can learn it later.", "source": "none", "notices": [msg], "learned": False}
        entry = self.kb.add(question, lesson["answer"], source)
        self.trainer.submit("learn", [entry.question])
        self.report({"question": question, "reason": reason, "confidence": round(conf, 4), "status": "learned",
                     "teacher": source, "via": lesson["via"], "entry": entry.id,
                     "cloud_report_id": lesson.get("report_id", ""), "answer_preview": lesson["answer"][:160]})
        msg = (f"🔔 Automatic report: Titan couldn't elaborate a response (confidence {conf:.2f}, {reason}). "
               f"Cloud call → learned from {who} → TITAN.md updated (entry #{entry.id}) → "
               f"training on it in the background.")
        if teacher_err:
            msg += f" (Cloud teacher unavailable: {teacher_err}; used the web instead.)"
        return {"reply": lesson["answer"], "source": source, "notices": [msg], "learned": True, "entry": entry.id}

    def answer(self, question: str, force_cloud: bool = False) -> dict:
        question = question.strip()
        hits = self.kb.search(question, k=3)
        best_score, best = hits[0] if hits else (0.0, None)
        relevant = best is not None and best_score >= self.cfg.sim_threshold
        if force_cloud:
            return {**self.learn(question, hits, "", 0.0, "lesson requested"), "confidence": 0.0}
        draft, conf = self.think(question, best.answer if relevant else "")
        too_short = len(draft) < 2
        garbled = self.kb.word_validity(draft) < self.cfg.min_word_validity
        if (not too_short and not garbled and conf >= self.cfg.conf_threshold
                and (relevant or conf >= self.cfg.high_conf)):
            return {"reply": draft, "source": "titan", "confidence": round(conf, 4), "notices": [],
                    "learned": False, "knowledge": best.id if relevant else None}
        reason = ("empty answer" if too_short else "garbled words" if garbled else
                  "low confidence" if conf < self.cfg.conf_threshold else "nothing about it in TITAN.md")
        if best is not None and best_score >= self.cfg.recall_threshold:
            # The answer is already in TITAN.md: no cloud call needed, just recall it and train on it.
            queued = self.trainer.submit("learn", [best.question])
            note = (f"🔔 Automatic report: Titan couldn't elaborate a response (confidence {conf:.2f}, {reason}) "
                    f"but TITAN.md already has it (entry #{best.id}); answered from the file"
                    + (" and training on it in the background." if queued else "."))
            return {"reply": best.answer, "source": "knowledge", "confidence": round(conf, 4),
                    "notices": [note], "learned": False, "knowledge": best.id}
        return {**self.learn(question, hits, draft, conf, reason), "confidence": round(conf, 4)}

    # -- chat entry point (handles slash commands) ----------------------------------
    def chat(self, message: str) -> dict:
        msg = message.strip()
        if not msg:
            return {"reply": "Say something!", "source": "titan", "notices": []}
        if not msg.startswith("/"):
            return self.answer(msg)
        cmd, _, arg = msg.partition(" ")
        arg = arg.strip()
        cmd = cmd.lower()
        try:
            if cmd in ("/help", "/?"):
                return {"reply": HELP, "source": "command", "notices": []}
            if cmd == "/status":
                return {"reply": json.dumps(self.info(), indent=2), "source": "command", "notices": []}
            if cmd == "/knowledge":
                return {"reply": f"TITAN.md has {len(self.kb.entries)} entries. Latest: "
                                 + "; ".join(e.question for e in self.kb.entries[-5:]),
                        "source": "command", "notices": []}
            if cmd == "/search":
                res = web_search(arg)
                lines = [f"• {r['title']}: {r['snippet'][:300]} {r['url']}" for r in res["results"]]
                return {"reply": "\n".join(lines) or "No results.", "source": "web", "notices": [], "search": res}
            if cmd == "/teach":
                if "|" not in arg:
                    return {"reply": "Usage: /teach question | answer", "source": "command", "notices": []}
                q, a = (s.strip() for s in arg.split("|", 1))
                e = self.kb.add(q, a, "user")
                self.trainer.submit("learn", [e.question])
                return {"reply": f"Thanks! Saved as entry #{e.id} in TITAN.md and training on it.",
                        "source": "command", "notices": [f"🔔 TITAN.md updated by you (entry #{e.id})."]}
            if cmd == "/learn":
                return self.answer(arg, force_cloud=True)
            if cmd == "/image":
                r = self.images.generate(arg)
                return {"reply": r.get("message") or f"Here's your image ({Path(r['file']).name}).",
                        "source": "perchance", "notices": [], "image": r}
            if cmd.startswith("/video"):
                backend = cmd.split(":", 1)[1] if ":" in cmd else "animate"
                r = self.videos.generate(arg, backend=backend)
                return {"reply": f"Video ready ({backend}): {Path(r['file']).name}", "source": backend,
                        "notices": r["notes"], "video": r}
            if cmd == "/tools":
                lines = []
                for url, cli in self.mcp.items():
                    for t in cli.list_tools():
                        lines.append(f"• {t['name']} ({url}): {t.get('description', '')[:120]}")
                return {"reply": "\n".join(lines) or "No MCP servers connected (set TITAN_MCP_SERVERS).",
                        "source": "mcp", "notices": []}
            if cmd == "/tool":
                name, _, js = arg.partition(" ")
                args = json.loads(js) if js.strip() else {}
                for url, cli in self.mcp.items():
                    if any(t["name"] == name for t in cli.list_tools()):
                        res = cli.call_tool(name, args)
                        text = "\n".join(c.get("text", f"[{c.get('type')}]") for c in res.get("content", []))
                        return {"reply": text or json.dumps(res), "source": f"mcp:{url}", "notices": []}
                return {"reply": f"No connected MCP server has a tool named {name!r}.", "source": "mcp", "notices": []}
        except Exception as ex:
            return {"reply": f"⚠️ {cmd} failed: {ex}", "source": "error", "notices": []}
        return {"reply": f"Unknown command {cmd}. Type /help.", "source": "command", "notices": []}

    def process_pending(self) -> list[dict]:
        """Retry questions that were queued while the cloud was unreachable."""
        done = []
        learned = {r["question"] for r in self.reports(10_000) if r.get("status") == "learned"}
        pending = [r for r in self.reports(10_000) if r.get("status") == "pending" and r["question"] not in learned]
        for q in dict.fromkeys(r["question"] for r in pending):
            done.append(self.learn(q, self.kb.search(q), "", 0.0, "queued while offline"))
        return done


# =============================================================================
# 9. MCP server (stdio + HTTP /mcp)
# =============================================================================

MCP_TOOLS = [
    {"name": "titan_chat", "description": "Chat with Titan-23M. It answers from its own model and TITAN.md, "
                                          "and learns from gpt-oss-120b when it can't.",
     "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}},
    {"name": "titan_generate_image", "description": "Generate an image with Titan's Perchance generator.",
     "inputSchema": {"type": "object", "properties": {"prompt": {"type": "string"},
                                                      "negative_prompt": {"type": "string"}}, "required": ["prompt"]}},
    {"name": "titan_generate_video", "description": "Generate a video on this device. Backends: "
                                                    + ", ".join(VIDEO_BACKENDS),
     "inputSchema": {"type": "object", "properties": {
         "prompt": {"type": "string"}, "backend": {"type": "string", "enum": list(VIDEO_BACKENDS)},
         "frames": {"type": "integer"}, "image_base64": {"type": "string"}}, "required": ["prompt"]}},
    {"name": "titan_download_video_generator", "description": "Download a video generator's weights to this device "
                                                              "(cogvideox, ltx, wan, modelscope or svd).",
     "inputSchema": {"type": "object", "properties": {"backend": {"type": "string"}}, "required": ["backend"]}},
    {"name": "titan_web_search", "description": "Search the web (DuckDuckGo instant answers + Wikipedia).",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "titan_teach", "description": "Teach Titan a question/answer pair (saved to TITAN.md, then trained).",
     "inputSchema": {"type": "object", "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
                     "required": ["question", "answer"]}},
    {"name": "titan_knowledge", "description": "Read Titan's TITAN.md knowledge file, or search it.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}},
]


def mcp_call_tool(brain: Brain, name: str, a: dict) -> dict:
    def text(t: str) -> dict:
        return {"content": [{"type": "text", "text": t}]}

    if name == "titan_chat":
        r = brain.chat(a.get("message", ""))
        return text("\n".join([r["reply"], *r.get("notices", [])]))
    if name == "titan_generate_image":
        r = brain.images.generate(a["prompt"], a.get("negative_prompt", ""))
        if r.get("ok"):
            data = Path(r["file"]).read_bytes()
            mime = mimetypes.guess_type(r["file"])[0] or "image/png"
            return {"content": [{"type": "image", "data": base64.b64encode(data).decode(), "mimeType": mime},
                                {"type": "text", "text": f"Saved to {r['file']}"}]}
        return text(json.dumps(r))
    if name == "titan_generate_video":
        img = base64.b64decode(a["image_base64"]) if a.get("image_base64") else None
        r = brain.videos.generate(a["prompt"], a.get("backend", "animate"), img, a.get("frames"))
        return text(json.dumps(r))
    if name == "titan_download_video_generator":
        return text(json.dumps(brain.videos.downloads.start(a["backend"])))
    if name == "titan_web_search":
        return text(json.dumps(web_search(a["query"]), ensure_ascii=False, indent=1))
    if name == "titan_teach":
        e = brain.kb.add(a["question"], a["answer"], "user")
        brain.trainer.submit("learn", [e.question])
        return text(f"Saved as entry #{e.id} in TITAN.md; training started.")
    if name == "titan_knowledge":
        if a.get("query"):
            return text("\n".join(e.to_md() for _, e in brain.kb.search(a["query"], 5)) or "Nothing found.")
        return text(brain.kb.text)
    raise KeyError(name)


def mcp_handle(brain: Brain, msg: dict) -> Optional[dict]:
    mid, method = msg.get("id"), msg.get("method", "")
    if mid is None:  # notification
        return None
    try:
        if method == "initialize":
            pv = (msg.get("params") or {}).get("protocolVersion") or "2025-06-18"
            result = {"protocolVersion": pv, "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": NAME, "version": __version__},
                      "instructions": "Titan-23M: chat, images (Perchance), on-device video, web search, teaching."}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": MCP_TOOLS}
        elif method == "tools/call":
            p = msg.get("params") or {}
            try:
                result = mcp_call_tool(brain, p.get("name", ""), p.get("arguments") or {})
            except KeyError as ex:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool or argument {ex}"}}
            except Exception as ex:
                result = {"content": [{"type": "text", "text": f"error: {ex}"}], "isError": True}
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}
    except Exception as ex:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(ex)}}


def run_mcp_stdio(brain: Brain) -> None:
    log("MCP server on stdio ready")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            out = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            if isinstance(msg, list):
                out = [r for r in (mcp_handle(brain, m) for m in msg) if r] or None
            else:
                out = mcp_handle(brain, msg)
        if out:
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()


# =============================================================================
# 10. HTTP API + browser UI
# =============================================================================

UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Titan-23M</title>
<meta name="description" content="Titan-23M: a free PyTorch AI (23M to 1.3B parameters) that learns what it doesn't know. Downloadable, or talk to it in the browser.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Unbounded:wght@500;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#f6f7fb;--panel:#fff;--ink:#141826;--muted:#5d6475;--line:#e3e6ee;--accent:#4f46e5;--accent-ink:#fff;
--me:#4f46e5;--me-ink:#fff;--bot:#eef0f6;--note:#fff6db;--note-ink:#6b4e00;--err:#b42318;
--grad-a:#4f46e5;--grad-b:#0891b2;--glow-a:rgba(79,70,229,.18);--glow-b:rgba(8,145,178,.14);--chip:#eceefe;--code:#0f1220;--code-ink:#dfe3f5}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0f1117;--panel:#171a23;--ink:#e8eaf2;
--muted:#9aa1b2;--line:#2a2f3c;--accent:#818cf8;--accent-ink:#0f1117;--me:#6366f1;--me-ink:#fff;--bot:#212532;
--note:#3a2f10;--note-ink:#ffd978;--err:#ff8a80;
--grad-a:#a5b4fc;--grad-b:#67e8f9;--glow-a:rgba(99,102,241,.28);--glow-b:rgba(34,211,238,.16);--chip:#1f2333;--code:#0b0d14;--code-ink:#dfe3f5}}
:root[data-theme="dark"]{--bg:#0f1117;--panel:#171a23;--ink:#e8eaf2;--muted:#9aa1b2;--line:#2a2f3c;--accent:#818cf8;
--accent-ink:#0f1117;--me:#6366f1;--me-ink:#fff;--bot:#212532;--note:#3a2f10;--note-ink:#ffd978;--err:#ff8a80;
--grad-a:#a5b4fc;--grad-b:#67e8f9;--glow-a:rgba(99,102,241,.28);--glow-b:rgba(34,211,238,.16);--chip:#1f2333;--code:#0b0d14;--code-ink:#dfe3f5}
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--ink);font:15px/1.55 Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.display{font-family:Unbounded,Inter,system-ui,sans-serif;letter-spacing:-.01em}
code,pre,.mono{font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
.app{max-width:900px;margin:0 auto;padding:16px;display:none;flex-direction:column;height:100vh;height:100dvh}
body.chatting .app{display:flex}body.chatting #home{display:none}
.app h1{font-family:Unbounded,Inter,sans-serif;font-weight:700}
.home-link{color:var(--muted);text-decoration:none;font-size:13px}.home-link:hover{color:var(--ink)}
.app header{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.app h1{font-size:20px;margin:0}.app h1 small{font-weight:400;color:var(--muted);font-size:13px;margin-left:6px}
.status{font-size:12px;color:var(--muted);margin-left:auto}
.app nav{display:flex;gap:4px;margin:12px 0;border-bottom:1px solid var(--line);overflow-x:auto}
.app nav button{background:none;border:0;border-bottom:2px solid transparent;color:var(--muted);padding:8px 12px;font:inherit;cursor:pointer}
.app nav button.on{color:var(--ink);border-color:var(--accent)}
.app>section{display:none;flex:1;min-height:0;flex-direction:column;gap:10px}.app>section.on{display:flex}
#log{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:8px;padding:4px 2px}
.msg{max-width:85%;padding:9px 13px;border-radius:14px;white-space:pre-wrap;word-wrap:break-word}
.me{align-self:flex-end;background:var(--me);color:var(--me-ink);border-bottom-right-radius:4px}
.bot{align-self:flex-start;background:var(--bot);border-bottom-left-radius:4px}
.note{align-self:center;background:var(--note);color:var(--note-ink);font-size:13px;max-width:95%;border-radius:8px}
.meta{display:block;font-size:11px;color:var(--muted);margin-top:4px}
.msg img,.msg video,.out img,.out video{max-width:100%;border-radius:8px;margin-top:6px;display:block}
form.row{display:flex;gap:8px}
input,textarea,select{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:9px 11px;min-width:0}
textarea{resize:vertical}input:focus,textarea:focus,select:focus{outline:2px solid var(--accent);outline-offset:-1px}
.row input{flex:1}
button.go{background:var(--accent);color:var(--accent-ink);border:0;border-radius:10px;padding:9px 16px;font:inherit;font-weight:600;cursor:pointer}
button.go:disabled{opacity:.5;cursor:wait}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;display:flex;flex-direction:column;gap:10px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}@media(max-width:560px){.grid{grid-template-columns:1fr}}
label{font-size:13px;color:var(--muted);display:flex;flex-direction:column;gap:4px}
.hint{font-size:13px;color:var(--muted)}.err{color:var(--err)}
pre{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;overflow:auto;white-space:pre-wrap;flex:1;margin:0;font-size:13px}
iframe{width:100%;background:#fff}
.dl-list{display:flex;flex-direction:column;gap:10px}
.dl-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px 12px;align-items:center;padding:10px 0;border-top:1px solid var(--line)}
.dl-row:first-child{border-top:0}
.dl-name{font-weight:600}.dl-meta{font-size:13px;color:var(--muted);overflow-wrap:anywhere}
.dl-bar{grid-column:1/-1;height:6px;border-radius:99px;background:var(--line);overflow:hidden}
.dl-bar i{display:block;height:100%;background:var(--accent);width:0;transition:width .4s}
.dl-row code{font-size:12px;overflow-wrap:anywhere}
button.ghost{background:none;border:1px solid var(--line);color:var(--ink);border-radius:10px;padding:7px 12px;font:inherit;font-size:14px;cursor:pointer}
button.ghost:disabled{opacity:.55;cursor:default}
a{color:var(--accent)}

/* ---------- landing page ---------- */
#home{overflow-x:hidden}
.wrap{max-width:1120px;margin:0 auto;padding:0 16px}
.topbar{position:sticky;top:0;z-index:5;backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  background:color-mix(in srgb,var(--bg) 78%,transparent);border-bottom:1px solid var(--line)}
.topbar .wrap{display:flex;align-items:center;gap:20px;height:60px}
.logo{font-family:Unbounded,sans-serif;font-weight:800;font-size:18px;color:var(--ink);text-decoration:none;display:flex;align-items:center;gap:8px}
.logo i{width:22px;height:22px;border-radius:7px;background:linear-gradient(135deg,var(--grad-a),var(--grad-b));display:inline-block}
.toplinks{display:flex;gap:18px;margin-left:auto}.toplinks a{color:var(--muted);text-decoration:none;font-size:14px}
.toplinks a:hover{color:var(--ink)}
@media(max-width:760px){.toplinks{display:none}.topbar .btn{margin-left:auto}}
.btn{display:inline-flex;align-items:center;gap:8px;border-radius:12px;padding:11px 18px;font:600 15px/1 Inter,sans-serif;
  text-decoration:none;cursor:pointer;border:1px solid transparent;white-space:nowrap}
.btn-primary{background:linear-gradient(135deg,var(--grad-a),var(--grad-b));color:#fff;box-shadow:0 8px 24px var(--glow-a)}
.btn-primary:hover{filter:brightness(1.08)}
.btn-ghost{background:var(--panel);color:var(--ink);border-color:var(--line)}.btn-ghost:hover{border-color:var(--accent)}
.btn-sm{padding:8px 14px;font-size:14px}
.hero{position:relative;padding:84px 0 56px;text-align:center}
.hero:before,.hero:after{content:"";position:absolute;z-index:-1;border-radius:50%;filter:blur(70px)}
.hero:before{width:520px;height:420px;left:50%;top:-60px;transform:translateX(-85%);background:var(--glow-a)}
.hero:after{width:460px;height:380px;left:50%;top:40px;transform:translateX(5%);background:var(--glow-b)}
.eyebrow{display:inline-flex;gap:8px;align-items:center;padding:6px 12px;border-radius:999px;background:var(--chip);
  color:var(--muted);font-size:13px;border:1px solid var(--line)}
.eyebrow b{color:var(--ink);font-weight:600}.eyebrow .dot{flex:none}
.dot{width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 0 3px rgba(34,197,94,.2)}
.hero h1{font-family:Unbounded,sans-serif;font-weight:800;font-size:clamp(44px,9vw,104px);line-height:1;margin:22px 0 18px;
  letter-spacing:-.03em;background:linear-gradient(120deg,var(--ink) 30%,var(--grad-a) 65%,var(--grad-b));
  -webkit-background-clip:text;background-clip:text;color:transparent}
.lead{font-size:clamp(17px,2.2vw,21px);color:var(--ink);max-width:720px;margin:0 auto 10px}
.quote{color:var(--muted);max-width:680px;margin:0 auto;font-size:15px}
.askbox{max-width:680px;margin:34px auto 0;display:flex;gap:8px;padding:8px;background:var(--panel);border:1px solid var(--line);
  border-radius:16px;box-shadow:0 12px 40px rgba(0,0,0,.08)}
.askbox input{flex:1;border:0;background:transparent;font-size:16px;padding:10px 12px}
.askbox input:focus{outline:none}
.askbox:focus-within{border-color:var(--accent)}
.chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin:14px auto 0;max-width:720px}
.chips button{background:var(--chip);color:var(--ink);border:1px solid var(--line);border-radius:999px;padding:6px 12px;
  font:13px Inter,sans-serif;cursor:pointer}.chips button:hover{border-color:var(--accent)}
.cta{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:26px}
.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:var(--line);border:1px solid var(--line);
  border-radius:16px;overflow:hidden;margin:8px 0 0}
.stat{background:var(--panel);padding:18px 16px;text-align:center}
.stat b{display:block;font-family:Unbounded,sans-serif;font-weight:700;font-size:22px}
.stat span{color:var(--muted);font-size:13px}
@media(max-width:760px){.stats{grid-template-columns:repeat(2,1fr)}.stat:last-child{grid-column:span 2}}
.block{padding:72px 0 8px}
.kicker{font:600 12px/1 Inter,sans-serif;letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}
.block h2{font-family:Unbounded,sans-serif;font-weight:700;font-size:clamp(26px,4vw,38px);line-height:1.15;margin:10px 0 12px;letter-spacing:-.02em}
.block .sub{color:var(--muted);max-width:680px;margin:0 0 28px}
.features{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
@media(max-width:980px){.features{grid-template-columns:repeat(2,1fr)}}@media(max-width:520px){.features{grid-template-columns:1fr}}
.feat{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;transition:transform .15s,border-color .15s}
.feat:hover{transform:translateY(-2px);border-color:var(--accent)}
.feat svg{width:36px;height:36px;padding:7px;border-radius:10px;background:var(--chip);color:var(--accent)}
.feat h3{font-size:16px;margin:14px 0 6px}.feat p{margin:0;color:var(--muted);font-size:14px}
.steps{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;counter-reset:s}
@media(max-width:980px){.steps{grid-template-columns:1fr}}
.step{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px}
.step:before{counter-increment:s;content:counter(s,decimal-leading-zero);font-family:Unbounded,sans-serif;font-weight:700;font-size:28px;
  background:linear-gradient(135deg,var(--grad-a),var(--grad-b));-webkit-background-clip:text;background-clip:text;color:transparent}
.step h3{font-size:15px;margin:8px 0 6px}.step p{margin:0;color:var(--muted);font-size:14px}
@media(max-width:980px){.step{display:grid;grid-template-columns:52px 1fr;column-gap:12px;padding:16px}
  .step:before{grid-row:span 2;font-size:24px;line-height:1.2}.step h3{margin:2px 0 4px}}
.notice-demo{margin-top:14px;background:var(--note);color:var(--note-ink);border-radius:12px;padding:12px 14px;font-size:14px}
.editions{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:760px){.editions{grid-template-columns:1fr}}
.ed{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:24px}
.ed h3{font-family:Unbounded,sans-serif;font-weight:700;font-size:20px;margin:0 0 4px}
.ed .tag{color:var(--muted);font-size:14px;margin:0 0 14px}
.ed ul{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:8px}
.ed li{padding-left:24px;position:relative;font-size:14px}
.ed li:before{content:"";position:absolute;left:0;top:6px;width:12px;height:12px;border-radius:4px;background:linear-gradient(135deg,var(--grad-a),var(--grad-b))}
.gen{display:grid;grid-template-columns:1fr 1.4fr;gap:14px}@media(max-width:860px){.gen{grid-template-columns:1fr}}
.table{width:100%;border-collapse:collapse;font-size:14px}
.table th,.table td{text-align:left;padding:10px 8px;border-bottom:1px solid var(--line);vertical-align:top}
.table th{color:var(--muted);font-weight:500;font-size:13px}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;background:var(--chip);color:var(--muted)}
.pill.on{background:rgba(34,197,94,.15);color:#16a34a}
.tscroll{overflow-x:auto}
.codebox{white-space:pre;flex:none;background:var(--code);color:var(--code-ink);border-radius:16px;padding:18px;overflow-x:auto;font-size:13px;line-height:1.6;margin:0;border:1px solid var(--line)}
.codebox .c{color:#8b93b5}.codebox .k{color:#93c5fd}.codebox .s{color:#86efac}
.endpoints{display:grid;grid-template-columns:repeat(2,1fr);gap:8px 18px;margin:0;padding:0;list-style:none}
@media(max-width:620px){.endpoints{grid-template-columns:1fr}}
.endpoints li{font-size:14px;color:var(--muted)}.endpoints code{color:var(--ink);font-size:13px}
.final{margin:80px 0 0;padding:56px 24px;text-align:center;border-radius:24px;border:1px solid var(--line);
  background:radial-gradient(600px 240px at 20% 0%,var(--glow-a),transparent),radial-gradient(500px 240px at 90% 100%,var(--glow-b),transparent),var(--panel)}
.final h2{font-family:Unbounded,sans-serif;font-size:clamp(24px,4vw,36px);margin:0 0 10px}
footer{color:var(--muted);font-size:13px;padding:36px 0 44px;text-align:center}
</style>
</head>
<body>
<div id="home">
<div class="topbar"><div class="wrap">
  <a class="logo" href="#" aria-label="Titan-23M home"><i></i>TITAN</a>
  <nav class="toplinks" aria-label="Sections"><a href="#features">Features</a><a href="#learn">How it learns</a><a href="#editions">Editions</a><a href="#generators">Generators</a><a href="#api">API</a></nav>
  <button class="btn btn-primary btn-sm" data-start>Start chatting</button>
</div></div>

<div class="wrap">
<header class="hero">
  <span class="eyebrow"><span class="dot"></span><span><b>Free &amp; open source</b> · PyTorch · <span id="hEdition">online</span></span></span>
  <h1>Titan-23M</h1>
  <p class="lead">A free AI, from 23 million up to 1.3 billion parameters, that learns what it doesn't know. It keeps its memory in <b>TITAN.md</b>, and when it can't answer, its teacher <b>gpt-oss-120b</b> on Cloudflare teaches it.</p>
  <p class="quote">Download it as one file or talk to it right here in the browser. You can connect it to an MCP server, or ask it to use a connector like web search.</p>
  <form class="askbox" id="heroForm"><input id="heroIn" autocomplete="off" placeholder="Ask Titan anything…" aria-label="Ask Titan"><button class="btn btn-primary">Ask</button></form>
  <div class="chips" id="chips">
    <button type="button">Who are you?</button><button type="button">What can you do?</button>
    <button type="button">How do you learn?</button><button type="button">/image a lighthouse at sunset</button>
    <button type="button">/search Alan Turing</button>
  </div>
  <div class="cta">
    <button class="btn btn-primary" data-start>Start chatting →</button>
    <a class="btn btn-ghost" href="https://github.com/vendemiaraul-cyber/Free-AI-Titan-23M" target="_blank" rel="noopener">Download titan.py</a>
  </div>
</header>

<div class="stats">
  <div class="stat"><b id="sParams">23.0M</b><span>parameters</span></div>
  <div class="stat"><b id="sFacts">—</b><span>facts in TITAN.md</span></div>
  <div class="stat"><b id="sShape">7 × 512</b><span>layers × width</span></div>
  <div class="stat"><b id="sCtx">1,536</b><span>bytes of context</span></div>
  <div class="stat"><b id="sTeacher">gpt-oss-120b</b><span>cloud teacher</span></div>
</div>

<section class="block" id="features">
  <div class="kicker">Features</div>
  <h2>Small model, lots of tricks</h2>
  <p class="sub">Everything comes in one Python file, or in the browser edition you're looking at.</p>
  <div class="features">
    <div class="feat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0-6 6c0 2.2 1.2 3.6 2.4 4.8.8.8 1.1 1.5 1.1 2.7V18h5v-1.5c0-1.2.3-1.9 1.1-2.7C16.8 12.6 18 11.2 18 9a6 6 0 0 0-6-6Z"/><path d="M10 21h4"/></svg>
      <h3>Learns from the cloud</h3><p>When it can't answer, it asks gpt-oss-120b on Cloudflare, posts an automatic report and trains on the answer.</p></div>
    <div class="feat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z"/><path d="M14 3v5h5M9 13h6M9 17h4"/></svg>
      <h3>TITAN.md memory</h3><p>Every fact it knows is a Markdown entry you can read, edit or teach yourself with /teach.</p></div>
    <div class="feat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="9" cy="9" r="2"/><path d="m21 15-5-5L5 21"/></svg>
      <h3>Perchance images</h3><p>Pictures come from Titan's own generator on perchance.org, built right into the Image tab.</p></div>
    <div class="feat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="5" width="15" height="14" rx="2"/><path d="m17 10 5-3v10l-5-3"/></svg>
      <h3>Video generators</h3><p>CogVideoX, LTX-Video, Wan 2.1, ModelScope and SVD run on your device. Animate also works online.</p></div>
    <div class="feat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
      <h3>Web search connector</h3><p>/search uses DuckDuckGo instant answers and Wikipedia, with no key needed.</p></div>
    <div class="feat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 7 4 12l5 5M15 7l5 5-5 5"/></svg>
      <h3>MCP server &amp; client</h3><p>Plug Titan into any MCP app, or connect it to other MCP servers and call their tools.</p></div>
    <div class="feat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 17V7a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v10"/><path d="M2 19h20M8 9l-2 2 2 2M16 9l2 2-2 2"/></svg>
      <h3>OpenAI-compatible API</h3><p>/v1/chat/completions works with existing SDKs, alongside image, video, teach and search endpoints.</p></div>
    <div class="feat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12M7 10l5 5 5-5"/><path d="M5 21h14"/></svg>
      <h3>One file, or the browser</h3><p>titan.py holds the model, memory, generators, API and UI. Or just chat online.</p></div>
  </div>
</section>

<section class="block" id="learn">
  <div class="kicker">How it learns</div>
  <h2>It knows when it doesn't know</h2>
  <p class="sub">Titan checks its own answer. It needs a high token probability, and the words must be real words from TITAN.md, not garble.</p>
  <div class="steps">
    <div class="step"><h3>Recall</h3><p>Finds the closest entry in TITAN.md and reads it as context.</p></div>
    <div class="step"><h3>Answer</h3><p>Titan's transformer (23M to 1.3B parameters) writes a draft byte by byte.</p></div>
    <div class="step"><h3>Check</h3><p>Low confidence, garbled words, or nothing in TITAN.md means it can't elaborate.</p></div>
    <div class="step"><h3>Cloud call</h3><p>gpt-oss-120b answers, and an automatic message reports it.</p></div>
    <div class="step"><h3>Train</h3><p>The lesson is added to TITAN.md and Titan fine-tunes on it in the background.</p></div>
  </div>
  <div class="notice-demo">🔔 Automatic report: Titan couldn't elaborate a response (confidence 0.21, nothing about it in TITAN.md). Cloud call → learned from gpt-oss-120b on Cloudflare → TITAN.md updated (entry #27) → training on it in the background.</div>
</section>

<section class="block" id="editions">
  <div class="kicker">Two editions</div>
  <h2>Download it, or talk to it online</h2>
  <p class="sub">Both editions share the same memory format, API and this interface.</p>
  <div class="editions">
    <div class="ed"><h3>Single file</h3><p class="tag">titan.py · runs on your device</p>
      <ul><li>The PyTorch model in four sizes, 23M to 1.3B, trained on TITAN.md</li><li>Learns in the background while you chat</li>
      <li>Drives the Perchance generator and saves your images</li><li>Downloads and runs every video generator locally</li>
      <li>REST API, OpenAI-compatible chat, and MCP over stdio or HTTP</li></ul></div>
    <div class="ed"><h3>In the browser</h3><p class="tag">Cloudflare edition · nothing to install</p>
      <ul><li>Chat straight from this page</li><li>TITAN.md stored in the cloud and shared with titan.py</li>
      <li>gpt-oss-120b teaches it whatever it doesn't know</li><li>Perchance images and browser-made animate videos</li>
      <li>REST API, OpenAI-compatible chat, and MCP over HTTP</li></ul></div>
  </div>
</section>

<section class="block" id="generators">
  <div class="kicker">Generators</div>
  <h2>Images and video</h2>
  <p class="sub">Images come from Perchance. Video generators run on your device, and only some of them also work online.</p>
  <div class="gen">
    <div class="ed"><h3>Image</h3><p class="tag">Titan's generator on perchance.org</p>
      <p style="margin:0 0 14px;font-size:14px">Type <code>/image</code> and a prompt in chat, or use the Image tab with the generator built in.</p>
      <a class="btn btn-ghost btn-sm" href="https://perchance.org/rg4m14ri" target="_blank" rel="noopener">Open perchance.org/rg4m14ri</a></div>
    <div class="ed"><h3>Video</h3><p class="tag">/video or the Video tab</p>
      <div class="tscroll"><table class="table">
        <tr><th>Generator</th><th>Makes</th><th>Where</th></tr>
        <tr><td><b>animate</b></td><td>image → video</td><td><span class="pill on">online</span> <span class="pill">device</span></td></tr>
        <tr><td><b>cogvideox</b></td><td>text → video</td><td><span class="pill">device</span></td></tr>
        <tr><td><b>ltx</b></td><td>text → video</td><td><span class="pill">device</span></td></tr>
        <tr><td><b>wan</b> 2.1</td><td>text → video</td><td><span class="pill">device</span></td></tr>
        <tr><td><b>modelscope</b></td><td>text → video</td><td><span class="pill">device</span></td></tr>
        <tr><td><b>svd</b></td><td>image → video</td><td><span class="pill">device</span></td></tr>
      </table></div></div>
  </div>
</section>

<section class="block" id="api">
  <div class="kicker">API</div>
  <h2>Build on Titan</h2>
  <p class="sub">Every feature is an endpoint. Chat is OpenAI-compatible, so existing SDKs work.</p>
  <div class="gen">
<pre class="codebox"><span class="c"># chat with Titan</span>
<span class="k">curl</span> <span id="apiBase">http://127.0.0.1:8023</span>/v1/chat \
  -H <span class="s">'Content-Type: application/json'</span> \
  -d <span class="s">'{"message": "What can you do?"}'</span>

<span class="c"># reply, confidence, source and any automatic reports</span>
{ <span class="s">"reply"</span>: <span class="s">"I can chat, answer questions…"</span>,
  <span class="s">"source"</span>: <span class="s">"titan"</span>, <span class="s">"confidence"</span>: 0.94,
  <span class="s">"notices"</span>: [] }</pre>
    <ul class="endpoints">
      <li><code>POST /v1/chat</code><br>chat, including slash commands</li>
      <li><code>POST /v1/chat/completions</code><br>OpenAI-compatible</li>
      <li><code>POST /v1/image</code><br>Perchance image</li>
      <li><code>POST /v1/video</code><br>video on your device</li>
      <li><code>POST /v1/teach</code><br>teach Titan an answer</li>
      <li><code>GET /v1/knowledge</code><br>read TITAN.md</li>
      <li><code>GET /v1/search?q=</code><br>web search connector</li>
      <li><code>POST /mcp</code><br>MCP JSON-RPC</li>
    </ul>
  </div>
</section>

<div class="final">
  <h2>Say hi to Titan</h2>
  <p class="quote" style="margin-bottom:22px">If it doesn't know the answer yet, it will once it's done learning.</p>
  <button class="btn btn-primary" data-start>Start chatting →</button>
</div>
<footer>Free-AI-Titan-23M · PyTorch · TITAN.md · gpt-oss-120b on Cloudflare · images by Perchance</footer>
</div>
</div>

<div class="app">
<header>
  <a class="home-link" href="#" id="homeLink">← Home</a>
  <h1>Titan-23M <small id="edition"></small></h1>
  <span class="status" id="status">connecting…</span>
</header>
<nav>
  <button class="on" data-tab="chat">Chat</button><button data-tab="image">Image</button>
  <button data-tab="video">Video</button><button data-tab="knowledge">Knowledge</button><button data-tab="settings">Settings</button>
</nav>

<section id="chat" class="on">
  <div id="log"></div>
  <form class="row" id="chatForm"><input id="chatIn" autocomplete="off" placeholder="Ask Titan… (/help for commands)"><button class="go">Send</button></form>
</section>

<section id="image">
  <div class="card">
    <label>Prompt<textarea id="imgPrompt" rows="3" placeholder="a lighthouse on a cliff at sunset, oil painting"></textarea></label>
    <label>Negative prompt<input id="imgNeg" placeholder="blurry, low quality"></label>
    <div><button class="go" id="imgGo">Generate with Perchance</button></div>
    <div class="hint" id="imgHint">Images come from your Perchance generator.</div>
  </div>
  <div class="out" id="imgOut"></div>
  <div class="card">
    <div class="hint">Your Perchance generator · <a id="pcLink" href="https://perchance.org/rg4m14ri" target="_blank" rel="noopener">open in a new tab</a></div>
    <iframe id="pcFrame" src="https://perchance.org/rg4m14ri"
      style="width:100%; max-width:700px; height:800px; border:0; border-radius:12px;"
      loading="lazy" allow="clipboard-write" title="Titan's Perchance image generator"></iframe>
  </div>
</section>

<section id="video">
  <div class="card">
    <label>Prompt<textarea id="vidPrompt" rows="2" placeholder="a paper boat drifting on a rainy street"></textarea></label>
    <div class="grid">
      <label>Generator<select id="vidBackend"></select></label>
      <label>Frames (blank = default)<input id="vidFrames" type="number" min="8" max="200"></label>
    </div>
    <label>Image to animate (optional)<input id="vidImage" type="file" accept="image/*"></label>
    <div><button class="go" id="vidGo">Generate video</button></div>
    <div class="hint" id="vidHint"></div>
  </div>
  <div class="out" id="vidOut"></div>
  <div class="card">
    <div><b>Download video generators</b> <span class="hint">· they run on your device, so their weights are saved there</span></div>
    <div id="dlList" class="dl-list"><span class="hint">loading…</span></div>
  </div>
</section>

<section id="knowledge">
  <div class="card">
    <div class="grid"><input id="tQ" placeholder="Question"><input id="tA" placeholder="Answer"></div>
    <div><button class="go" id="teachGo">Teach Titan</button> <span class="hint" id="teachHint"></span></div>
  </div>
  <pre id="md">loading TITAN.md…</pre>
</section>

<section id="settings">
  <div class="card">
    <label>API key (only if the server requires one)<input id="key" type="password" autocomplete="off"></label>
    <label>Theme<select id="theme"><option value="">System</option><option value="light">Light</option><option value="dark">Dark</option></select></label>
    <div class="hint">API: <code>POST /v1/chat</code>, <code>/v1/chat/completions</code>, <code>/v1/image</code>, <code>/v1/video</code>, <code>/v1/teach</code>, <code>GET /v1/knowledge</code>, <code>/v1/search?q=</code>, MCP at <code>/mcp</code>.</div>
  </div>
</section>
</div>
<script>
const $ = s => document.querySelector(s);
const store = {get(k){try{return localStorage.getItem(k)||""}catch(e){return ""}},set(k,v){try{localStorage.setItem(k,v)}catch(e){}}};
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function headers(json=true){const h={}; if(json)h["Content-Type"]="application/json"; const k=store.get("titanKey"); if(k)h["Authorization"]="Bearer "+k; return h;}
async function api(path, body){
  const r = await fetch(path, body===undefined?{headers:headers(false)}:{method:"POST",headers:headers(),body:JSON.stringify(body)});
  const ct = r.headers.get("content-type")||"";
  const data = ct.includes("json") ? await r.json() : await r.text();
  if(!r.ok) throw new Error((data&&data.error)||data||r.statusText);
  return data;
}
// tabs + settings
document.querySelectorAll(".app nav button").forEach(b=>b.onclick=()=>{
  document.querySelectorAll(".app nav button,.app>section").forEach(x=>x.classList.remove("on"));
  b.classList.add("on"); $("#"+b.dataset.tab).classList.add("on");
  if(b.dataset.tab==="knowledge") loadMd();
});
$("#key").value=store.get("titanKey"); $("#key").onchange=e=>store.set("titanKey",e.target.value);
$("#theme").value=store.get("titanTheme"); applyTheme();
$("#theme").onchange=e=>{store.set("titanTheme",e.target.value);applyTheme()};
function applyTheme(){const t=store.get("titanTheme"); if(t)document.documentElement.dataset.theme=t; else delete document.documentElement.dataset.theme;}

// landing page <-> app
function openApp(tab){document.body.classList.add("chatting"); if(location.hash!=="#chat") history.pushState(null,"","#chat");
  if(tab) document.querySelector(`nav button[data-tab="${tab}"]`).click(); setTimeout(()=>$("#chatIn").focus(),0);}
function showHome(){document.body.classList.remove("chatting"); if(location.hash==="#chat") history.pushState(null,"",location.pathname); window.scrollTo(0,0);}
document.querySelectorAll("[data-start]").forEach(b=>b.onclick=()=>openApp("chat"));
$("#homeLink").onclick=e=>{e.preventDefault();showHome()};
window.addEventListener("popstate",()=>location.hash==="#chat"?document.body.classList.add("chatting"):document.body.classList.remove("chatting"));
if(location.hash==="#chat") document.body.classList.add("chatting");
$("#heroForm").onsubmit=e=>{e.preventDefault(); const m=$("#heroIn").value.trim(); $("#heroIn").value=""; openApp("chat"); if(m) send(m);};
document.querySelectorAll("#chips button").forEach(b=>b.onclick=()=>{openApp("chat"); send(b.textContent);});
$("#apiBase").textContent=location.origin;

let EDITION="";
api("/v1/health").then(h=>{
  EDITION=h.edition||""; $("#edition").textContent=EDITION;
  $("#hEdition").textContent=EDITION==="Cloudflare"?"in your browser":"on your device";
  if(h.parameters) $("#sParams").textContent = h.parameters>=1e9 ? (h.parameters/1e9).toFixed(2)+"B" : (h.parameters/1e6).toFixed(1)+"M";
  if(h.config){ $("#sShape").textContent=h.config.n_layer+" × "+h.config.d_model; $("#sCtx").textContent=h.config.ctx.toLocaleString("en-US"); }
  if(h.knowledge_entries!=null) $("#sFacts").textContent=h.knowledge_entries;
  if(h.teacher) $("#sTeacher").textContent=h.teacher;
  $("#status").textContent=[h.parameters?(h.parameters/1e6).toFixed(1)+"M params":"", h.knowledge_entries+" facts",
    "teacher: "+(h.teacher||"gpt-oss-120b")].filter(Boolean).join(" · ");
}).catch(e=>$("#status").textContent="offline: "+e.message);

// chat
function add(cls, html, meta){const d=document.createElement("div"); d.className="msg "+cls; d.innerHTML=html+(meta?`<span class="meta">${esc(meta)}</span>`:""); $("#log").appendChild(d); $("#log").scrollTop=1e9; return d;}
function media(r){
  let h="";
  const img=r.image, vid=r.video;
  if(img&&img.image_url) h+=`<img src="${esc(img.image_url)}" alt="">`;
  else if(img&&img.url) h+=`<br><a href="${esc(img.url)}" target="_blank" rel="noopener">Open your Perchance generator</a>`;
  if(vid&&vid.video_url) h+=vid.video_url.endsWith(".gif")?`<img src="${esc(vid.video_url)}" alt="">`:`<video src="${esc(vid.video_url)}" controls autoplay loop muted playsinline></video>`;
  return h;
}
add("bot","Hi, I'm Titan. Ask me anything. When I don't know, my teacher gpt-oss-120b on Cloudflare teaches me and I write it into TITAN.md. Type /help for commands.");
$("#chatForm").onsubmit=e=>{e.preventDefault(); const m=$("#chatIn").value.trim(); if(!m)return; $("#chatIn").value=""; send(m);};
async function send(m){
  add("me",esc(m)); const wait=add("bot","…");
  try{
    const r=await api("/v1/chat",{message:m});
    const conf = typeof r.confidence==="number" ? ` · confidence ${r.confidence.toFixed(2)}` : "";
    wait.innerHTML=esc(r.reply)+media(r)+`<span class="meta">${esc(r.source||"")}${conf}</span>`;
    (r.notices||[]).forEach(n=>add("note",esc(n)));
  }catch(err){wait.innerHTML=`<span class="err">${esc(err.message)}</span>`}
};

// image
api("/v1/image/status").then(s=>{ if(s.generator_url && s.generator_url!==$("#pcFrame").getAttribute("src")){
  $("#pcFrame").src=s.generator_url; $("#pcLink").href=s.generator_url; } }).catch(()=>{});
$("#imgGo").onclick=async()=>{
  const p=$("#imgPrompt").value.trim(); if(!p)return; const b=$("#imgGo"); b.disabled=true; $("#imgHint").textContent="Generating… (this can take a minute)";
  try{
    const r=await api("/v1/image",{prompt:p,negative_prompt:$("#imgNeg").value});
    if(r.image_url){$("#imgOut").innerHTML=`<img src="${esc(r.image_url)}" alt="${esc(p)}"><p class="hint"><a href="${esc(r.image_url)}" download>Download</a></p>`;$("#imgHint").textContent="Done.";}
    else if(r.url){ let copied=false; try{await navigator.clipboard.writeText(p); copied=true}catch(e){}
      $("#imgHint").innerHTML=esc(r.message||"")+(copied?" The prompt is on your clipboard.":"");
      $("#imgOut").innerHTML=`<p class="hint">Prompt to paste: <b>${esc(p)}</b></p>`; $("#pcFrame").scrollIntoView({behavior:"smooth"});}
    else $("#imgHint").textContent=r.message||"No image.";
  }catch(err){$("#imgHint").innerHTML=`<span class="err">${esc(err.message)}</span>`} finally{b.disabled=false}
};

// video
let BACKENDS=[];
api("/v1/video/backends").then(r=>{
  BACKENDS=r.backends||[];
  $("#vidBackend").innerHTML=`<option value="browser">animate — in this browser (works online)</option>`+
    BACKENDS.map(b=>`<option value="${esc(b.name)}" ${b.available?"":"disabled"}>${esc(b.name)} — ${esc(b.where)}${b.available?"":" (not installed)"}</option>`).join("");
  showVidHint(); renderDownloads();
}).catch(()=>{$("#vidBackend").innerHTML=`<option value="browser">animate — in this browser (works online)</option>`; showVidHint(); renderDownloads();});

// downloads of the on-device video generators
const gb = n => n==null ? "" : (n/1e9).toFixed(1)+" GB";
let dlTimer=null;
function renderDownloads(){
  const heavy = BACKENDS.filter(b=>b.model);
  if(!heavy.length || !heavy.some(b=>b.download)){   // the Cloudflare edition can't run these
    $("#dlList").innerHTML=`<span class="hint">These generators run on your own device. Get <a href="https://github.com/vendemiaraul-cyber/Free-AI-Titan-23M" target="_blank" rel="noopener">titan.py</a>, then run <code>python titan.py download</code> or open this tab in <code>python titan.py serve</code>.</span>`;
    return;
  }
  const needPip = heavy.find(b=>b.missing && b.missing.length);
  const head = needPip ? `<div class="hint">First install the packages they need: <code>python titan.py download --deps</code> (or <code>${esc(needPip.pip||"")}</code>), then restart <code>titan.py serve</code>.</div>` : "";
  $("#dlList").innerHTML = head + heavy.map(b=>{
    const d=b.download||{}, st=d.state||"not downloaded";
    const pct = d.bytes_total ? Math.round(100*(d.bytes_done||0)/d.bytes_total) : 0;
    let meta = `${esc(b.kind)} · ${esc(b.model)}${b.download_gb?` · ~${b.download_gb} GB`:""}${b.gated?" · needs HF_TOKEN (gated)":""}`, btn = "";
    if(b.missing && b.missing.length){ btn=`<button class="ghost" disabled>Download</button>`; }
    else if(st==="downloading"){ meta += `<br>Downloading… ${pct}% · ${gb(d.bytes_done)} of ${gb(d.bytes_total)||"?"} · file ${d.files_done||0}/${d.files_total||"?"}`; btn=`<button class="ghost" disabled>${pct}%</button>`; }
    else if(st==="downloaded"){ meta += `<br>✓ Downloaded (${gb(d.bytes_total)}). Pick it in the generator list above.`; btn=`<button class="ghost" disabled>Ready</button>`; }
    else if(st==="error"){ meta += `<br><span class="err">${esc(d.error||"failed")}</span>`; btn=`<button class="ghost" data-dl="${esc(b.name)}">Retry</button>`; }
    else { btn=`<button class="ghost" data-dl="${esc(b.name)}">Download</button>`; }
    const bar = st==="downloading" ? `<div class="dl-bar"><i style="width:${pct}%"></i></div>` : "";
    return `<div class="dl-row"><div><span class="dl-name">${esc(b.name)}</span> <span class="dl-meta">${meta}</span></div>${btn}${bar}</div>`;
  }).join("");
  document.querySelectorAll("[data-dl]").forEach(x=>x.onclick=async()=>{
    x.disabled=true; x.textContent="Starting…";
    try{ await api("/v1/video/download",{backend:x.dataset.dl}); }catch(e){ x.textContent="Error"; alert(e.message); }
    refreshDownloads();
  });
  clearTimeout(dlTimer);
  if(heavy.some(b=>(b.download||{}).state==="downloading")) dlTimer=setTimeout(refreshDownloads,2000);
}
async function refreshDownloads(){ try{ const r=await api("/v1/video/backends"); BACKENDS=r.backends||[]; }catch(e){} renderDownloads(); }
$("#vidBackend").onchange=showVidHint;
function showVidHint(){const v=$("#vidBackend").value; const b=BACKENDS.find(x=>x.name===v);
  $("#vidHint").textContent = v==="browser" ? "Runs right here: animates your image (or a title card) into a WebM clip." : (b? b.description+(b.warning?" ⚠ "+b.warning:"") : "");}
function readFile(f){return new Promise((res,rej)=>{const r=new FileReader();r.onload=()=>res(r.result);r.onerror=rej;r.readAsDataURL(f);});}
$("#vidGo").onclick=async()=>{
  const p=$("#vidPrompt").value.trim(), be=$("#vidBackend").value, f=$("#vidImage").files[0], btn=$("#vidGo");
  btn.disabled=true; $("#vidHint").textContent="Generating… heavy generators can take minutes.";
  try{
    const dataUrl = f ? await readFile(f) : null;
    if(be==="browser"){ const url=await browserAnimate(dataUrl,p,+$("#vidFrames").value||72);
      $("#vidOut").innerHTML=`<video src="${url}" controls autoplay loop muted playsinline></video><p class="hint"><a href="${url}" download="titan-animate.webm">Download</a></p>`; $("#vidHint").textContent="Done.";}
    else{ const r=await api("/v1/video",{prompt:p,backend:be,frames:+$("#vidFrames").value||null,image_base64:dataUrl?dataUrl.split(",")[1]:null});
      const u=r.video_url; $("#vidOut").innerHTML=(u.endsWith(".gif")?`<img src="${esc(u)}" alt="">`:`<video src="${esc(u)}" controls autoplay loop muted playsinline></video>`)+`<p class="hint"><a href="${esc(u)}" download>Download</a> ${esc((r.notes||[]).join(" "))}</p>`; $("#vidHint").textContent="Done.";}
  }catch(err){$("#vidHint").innerHTML=`<span class="err">${esc(err.message)}</span>`} finally{btn.disabled=false}
};
async function browserAnimate(dataUrl, prompt, frames){
  const W=768,H=432,fps=24,c=document.createElement("canvas"); c.width=W; c.height=H; const g=c.getContext("2d");
  let img=null;
  if(dataUrl){img=new Image(); img.src=dataUrl; await img.decode();}
  const draw=t=>{const e=t*t*(3-2*t), z=1+0.22*e;
    if(img){const s=Math.max(W/img.width,H/img.height)*z, w=img.width*s, h=img.height*s;
      g.drawImage(img,(W-w)/2+(w-W)*0.25*(e-0.5),(H-h)/2-(h-H)*0.15*(e-0.5),w,h);}
    else{const gr=g.createLinearGradient(0,0,0,H); gr.addColorStop(0,"#141a3c"); gr.addColorStop(1,"#3c4ab4"); g.fillStyle=gr; g.fillRect(0,0,W,H);
      g.save(); g.translate(W/2,H/2); g.scale(z,z); g.fillStyle="#f0f0ff"; g.font="600 28px system-ui"; g.textAlign="center";
      const words=(prompt||"Titan-23M").split(" "); let line="",lines=[]; for(const w of words){if((line+" "+w).length>34){lines.push(line);line=w}else line=(line+" "+w).trim()} lines.push(line);
      lines.slice(0,5).forEach((l,i)=>g.fillText(l,0,(i-(Math.min(lines.length,5)-1)/2)*36)); g.restore();}};
  draw(0);
  const stream=c.captureStream(fps), rec=new MediaRecorder(stream,{mimeType:MediaRecorder.isTypeSupported("video/webm;codecs=vp9")?"video/webm;codecs=vp9":"video/webm"});
  const chunks=[]; rec.ondataavailable=e=>e.data.size&&chunks.push(e.data);
  const done=new Promise(r=>rec.onstop=r); rec.start();
  for(let i=0;i<frames;i++){draw(i/(frames-1)); await new Promise(r=>setTimeout(r,1000/fps));}
  rec.stop(); await done; return URL.createObjectURL(new Blob(chunks,{type:"video/webm"}));
}

// knowledge
async function loadMd(){try{$("#md").textContent=await api("/v1/knowledge")}catch(e){$("#md").textContent=e.message}}
$("#teachGo").onclick=async()=>{const q=$("#tQ").value.trim(),a=$("#tA").value.trim(); if(!q||!a)return;
  try{const r=await api("/v1/teach",{question:q,answer:a}); $("#teachHint").textContent=r.message||"Saved."; $("#tQ").value=$("#tA").value=""; loadMd();}
  catch(e){$("#teachHint").innerHTML=`<span class="err">${esc(e.message)}</span>`}};
</script>
</body>
</html>
"""


class TitanHTTP(BaseHTTPRequestHandler):
    brain: Brain  # set by serve()
    server_version = f"Titan/{__version__}"

    def log_message(self, fmt, *args):
        log("http", self.address_string(), fmt % args)

    # -- helpers ---------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str, extra: Optional[dict] = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, Mcp-Session-Id")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client went away

    def _json(self, obj: Any, code: int = 200, extra: Optional[dict] = None):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", extra)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 50 * 1024 * 1024:
            raise ValueError("request too large")
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw or b"{}")

    def _authorized(self) -> bool:
        key = self.brain.cfg.local_key
        if not key:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {key}"

    def _file_url(self, path: str) -> str:
        return "/outputs/" + urllib.parse.quote(Path(path).name)

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def do_HEAD(self):
        self.do_GET()

    # -- routes ----------------------------------------------------------------
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        p, qs = u.path, urllib.parse.parse_qs(u.query)
        b = self.brain
        try:
            if p in ("/", "/index.html"):
                return self._send(200, UI_HTML.encode(), "text/html; charset=utf-8")
            if p.startswith("/outputs/"):
                f = b.cfg.out_dir / Path(urllib.parse.unquote(p[len("/outputs/"):])).name
                if not f.is_file():
                    return self._json({"error": "not found"}, 404)
                return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
            if p == "/v1/health":
                return self._json({"ok": True, "name": NAME, "version": __version__, "edition": "titan.py",
                                   "size": b.cfg.size, "parameters": b.model.num_params(),
                                   "config": asdict(b.cfg.model), "knowledge_entries": len(b.kb.entries),
                                   "teacher": b.cfg.teacher_model.split("/")[-1], "teacher_mode": b.teacher.mode,
                                   "training": b.trainer.status()})
            if not self._authorized():
                return self._json({"error": "unauthorized"}, 401)
            if p == "/v1/model":
                return self._json(b.info())
            if p == "/v1/knowledge":
                return self._send(200, b.kb.text.encode(), "text/markdown; charset=utf-8")
            if p == "/v1/knowledge/entries":
                q = (qs.get("q") or [""])[0]
                ents = [e for _, e in b.kb.search(q, 10)] if q else b.kb.entries
                return self._json({"entries": [asdict(e) for e in ents]})
            if p == "/v1/search":
                return self._json(web_search((qs.get("q") or [""])[0]))
            if p == "/v1/video/backends":
                return self._json({"backends": b.videos.backends()})
            if p == "/v1/video/download":
                return self._json({"downloads": {n: b.videos.downloads.status(n) for n in VIDEO_BACKENDS}})
            if p == "/v1/image/status":
                return self._json(b.images.status())
            if p == "/v1/reports":
                return self._json({"reports": b.reports()})
            if p == "/mcp":
                return self._json({"error": "use POST for MCP JSON-RPC"}, 405, {"Allow": "POST"})
            return self._json({"error": "not found"}, 404)
        except Exception as ex:
            log(traceback.format_exc())
            return self._json({"error": str(ex)}, 500)

    def do_POST(self):
        p = urllib.parse.urlparse(self.path).path
        b = self.brain
        if not self._authorized():
            return self._json({"error": "unauthorized"}, 401)
        try:
            body = self._body()
            if p == "/v1/chat":
                r = b.chat(str(body.get("message", "")))
                for k in ("image", "video"):
                    if r.get(k) and r[k].get("file"):
                        r[k][k + "_url"] = self._file_url(r[k]["file"])
                return self._json(r)
            if p == "/v1/chat/completions":  # OpenAI-compatible
                msgs = body.get("messages") or []
                last = next((m for m in reversed(msgs) if m.get("role") == "user"), {"content": ""})
                content = last.get("content")
                if isinstance(content, list):
                    content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                r = b.chat(str(content))
                return self._json({
                    "id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion", "created": int(time.time()),
                    "model": "titan-23m", "choices": [{"index": 0, "finish_reason": "stop",
                                                       "message": {"role": "assistant", "content": r["reply"]}}],
                    "usage": {"prompt_tokens": len(encode(str(content))), "completion_tokens": len(encode(r["reply"])),
                              "total_tokens": len(encode(str(content))) + len(encode(r["reply"]))},
                    "titan": {k: v for k, v in r.items() if k != "reply"}})
            if p == "/v1/learn":
                return self._json(b.answer(str(body.get("question", "")), force_cloud=True))
            if p == "/v1/teach":
                q, a = str(body.get("question", "")).strip(), str(body.get("answer", "")).strip()
                if not q or not a:
                    return self._json({"error": "question and answer are required"}, 400)
                e = b.kb.add(q, a, body.get("source") or "user")
                b.trainer.submit("learn", [e.question])
                return self._json({"ok": True, "entry": asdict(e), "message": f"Saved as entry #{e.id}; training started."})
            if p == "/v1/knowledge":  # merge another TITAN.md (e.g. from the cloud)
                n = b.kb.merge_markdown(str(body.get("markdown", "")))
                return self._json({"ok": True, "added": n, "entries": len(b.kb.entries)})
            if p == "/v1/image":
                r = b.images.generate(str(body.get("prompt", "")), str(body.get("negative_prompt", "")))
                if r.get("file"):
                    r["image_url"] = self._file_url(r["file"])
                return self._json(r)
            if p == "/v1/video":
                img = base64.b64decode(body["image_base64"]) if body.get("image_base64") else None
                r = b.videos.generate(str(body.get("prompt", "")), body.get("backend") or "animate", img,
                                      body.get("frames"), int(body.get("fps") or 0) or 12, body.get("steps"),
                                      body.get("seed"), str(body.get("negative_prompt", "")))
                r["video_url"] = self._file_url(r["file"])
                return self._json(r)
            if p == "/v1/video/download":  # start downloading a video generator's weights
                name = str(body.get("backend", ""))
                return self._json({"backend": name, "download": b.videos.downloads.start(name)})
            if p == "/v1/train":
                ok = b.trainer.submit("bootstrap", steps=int(body.get("steps") or b.cfg.bootstrap_steps))
                return self._json({"ok": ok, "training": b.trainer.status()})
            if p == "/mcp":
                if isinstance(body, list):
                    out = [r for r in (mcp_handle(b, m) for m in body) if r]
                else:
                    out = mcp_handle(b, body)
                if not out:
                    return self._send(202, b"", "text/plain")
                sid = self.headers.get("Mcp-Session-Id") or uuid.uuid4().hex
                return self._json(out, extra={"Mcp-Session-Id": sid})
            return self._json({"error": "not found"}, 404)
        except ValueError as ex:
            return self._json({"error": str(ex)}, 400)
        except Exception as ex:
            log(traceback.format_exc())
            return self._json({"error": str(ex)}, 500)


def make_server(brain: Brain, host: str = "127.0.0.1", port: int = 8023) -> ThreadingHTTPServer:
    handler = type("Handler", (TitanHTTP,), {"brain": brain})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    return srv


# =============================================================================
# 11. CLI
# =============================================================================

def cmd_sync(brain: Brain) -> None:
    cfg = brain.cfg
    if not cfg.cloud_url:
        log("TITAN_CLOUD_URL isn't set; only retrying queued questions")
    else:
        h = {"Authorization": f"Bearer {cfg.cloud_key}"} if cfg.cloud_key else {}
        req = urllib.request.Request(cfg.cloud_url + "/v1/knowledge", headers={"User-Agent": USER_AGENT, **h})
        with urllib.request.urlopen(req, timeout=30) as r:
            cloud_md = r.read().decode()
        n = brain.kb.merge_markdown(cloud_md)
        log(f"pulled {n} new entries from the cloud")
        r = http_json(cfg.cloud_url + "/v1/knowledge", {"markdown": brain.kb.text}, headers=h)
        log(f"pushed TITAN.md to the cloud: {r}")
        if n:
            brain.trainer.submit("bootstrap", steps=max(50, 10 * n))
    for res in brain.process_pending():
        print(res["notices"][0] if res.get("notices") else res)


def cmd_download(names: list[str], deps: bool) -> None:
    """Download video generator weights (and optionally their pip packages) with progress."""
    import subprocess
    downloadable = [n for n, sp in VIDEO_BACKENDS.items() if "model" in sp]
    if deps:
        print("installing:", " ".join(VIDEO_PIP))
        subprocess.check_call([sys.executable, "-m", "pip", "install", *VIDEO_PIP])
    if not _has("huggingface_hub"):
        raise SystemExit("huggingface_hub is missing: run  python titan.py download --deps  (or pip install "
                         + " ".join(VIDEO_PIP) + ")")
    cfg = Config.from_env()
    dl = VideoDownloader(cfg)
    if not names:
        print("Video generators you can download (python titan.py download <name>|all):")
        for n in downloadable:
            sp, st = VIDEO_BACKENDS[n], dl.status(n)
            try:
                size = f"{sum(sz for _, sz in dl.files(n)) / 1e9:.1f} GB"
            except Exception as ex:  # offline etc.
                size = f"size unknown ({type(ex).__name__})"
            print(f"  {n:11} {sp['kind']:15} {sp['model']:48} {size:>9}  {st['state']}")
        return
    if names == ["all"]:
        names = downloadable
    for n in names:
        if n not in downloadable:
            raise SystemExit(f"unknown video generator {n!r}; choose from {', '.join(downloadable)} or all")
        files = dl.files(n)
        total = sum(sz for _, sz in files)
        print(f"downloading {n} ({VIDEO_BACKENDS[n]['model']}): {len(files)} files, {total / 1e9:.1f} GB")
        t0 = time.time()

        def on_file(path, i, count, done):
            pct = 100 * done / total if total else 100
            print(f"  [{i}/{count}] {pct:5.1f}%  {path}  ({time.time() - t0:.0f}s)", flush=True)
        dl.download(n, on_file)
        print(f"  ✓ {n} is ready: python titan.py video \"your prompt\" --backend {n}")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="titan.py", description=f"{NAME}: single-file AI (PyTorch).")
    sub = ap.add_subparsers(dest="cmd")
    sized = argparse.ArgumentParser(add_help=False)
    sized.add_argument("--size", help="model size: auto, " + ", ".join(MODEL_PRESETS) + " (default: $TITAN_SIZE or auto)")
    _add = sub.add_parser
    sub.add_parser = lambda *a, **k: _add(*a, parents=[sized], **k)  # every command accepts --size
    s = sub.add_parser("serve", help="HTTP API + browser UI")
    s.add_argument("--host", default=os.environ.get("TITAN_HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.environ.get("TITAN_PORT", "8023")))
    sub.add_parser("chat", help="chat in the terminal")
    a = sub.add_parser("ask", help="ask one question")
    a.add_argument("question", nargs="+")
    t = sub.add_parser("train", help="train on TITAN.md")
    t.add_argument("--steps", type=int, default=None)
    t.add_argument("--lr", type=float, default=None)
    sub.add_parser("mcp", help="MCP server over stdio")
    i = sub.add_parser("image", help="generate an image with your Perchance generator")
    i.add_argument("prompt", nargs="+")
    v = sub.add_parser("video", help="generate a video on this device")
    v.add_argument("prompt", nargs="+")
    v.add_argument("--backend", default="animate", choices=list(VIDEO_BACKENDS))
    v.add_argument("--image", help="image file to animate")
    v.add_argument("--frames", type=int)
    sub.add_parser("sync", help="sync TITAN.md with the cloud and learn queued questions")
    sub.add_parser("info", help="model size and status")
    sub.add_parser("sizes", help="list the model sizes and what hardware they need")
    d = sub.add_parser("download", help="download video generators to this device")
    d.add_argument("backends", nargs="*", help="cogvideox, ltx, wan, modelscope, svd, or all (default: list them)")
    d.add_argument("--deps", action="store_true", help="also pip install the packages video generators need")
    args = ap.parse_args(argv)
    cmd = args.cmd or "chat"

    if getattr(args, "size", None):
        os.environ["TITAN_SIZE"] = args.size
    if cmd == "sizes":
        chosen = resolve_size(os.environ.get("TITAN_SIZE", "auto"))
        print(f"{'size':6} {'params':>8} {'layers':>6} {'width':>6} {'context':>7} {'training memory':>16}  needs")
        for k, p in MODEL_PRESETS.items():
            mark = "  <- selected" if k == chosen else ""
            print(f"{k:6} {p['params']:>8} {p['n_layer']:>6} {p['d_model']:>6} {p['ctx']:>7} "
                  f"{'~' + str(p['train_gb']) + ' GB':>16}  {p['hardware']}{mark}")
        print("\nPick one with --size or TITAN_SIZE (default: auto = the biggest this machine can train).")
        return
    if cmd == "download":
        cmd_download(args.backends, args.deps)
        return

    cfg = Config.from_env()
    if cmd in ("train", "info", "image", "video", "mcp", "ask", "sync"):
        cfg.auto_train = False  # these commands shouldn't spawn background bootstrap training
    brain = Brain(cfg)

    if cmd == "info":
        print(json.dumps(brain.info(), indent=2))
    elif cmd == "train":
        steps = args.steps or cfg.bootstrap_steps
        print(f"training {brain.model.num_params()/1e6:.2f}M params on {len(brain.kb.entries)} entries, "
              f"{steps} steps on {brain.device}")
        t0 = time.time()

        def on_step(i, loss):
            if i % 10 == 0 or i == steps - 1:
                print(f"  step {i+1:5d}/{steps}  loss {loss:.4f}  ({time.time()-t0:.0f}s)", flush=True)

        train_steps(brain.model, brain.kb.entries, steps, args.lr or cfg.lr, cfg.batch_size, on_step=on_step)
        brain.trained_steps += steps
        brain.save()
        print(f"saved {cfg.ckpt_path}")
    elif cmd == "serve":
        srv = make_server(brain, args.host, args.port)
        print(f"Titan ({cfg.size}, {brain.model.num_params()/1e6:.1f}M params) is listening on "
              f"http://{args.host}:{args.port}  (UI at /, API at /v1, MCP at /mcp)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    elif cmd == "mcp":
        run_mcp_stdio(brain)
    elif cmd == "ask":
        r = brain.chat(" ".join(args.question))
        print(r["reply"])
        for n in r.get("notices", []):
            print(n)
    elif cmd == "image":
        print(json.dumps(brain.images.generate(" ".join(args.prompt)), indent=2))
    elif cmd == "video":
        img = Path(args.image).read_bytes() if args.image else None
        print(json.dumps(brain.videos.generate(" ".join(args.prompt), args.backend, img, args.frames), indent=2))
    elif cmd == "sync":
        cmd_sync(brain)
    else:  # chat
        print(f"{NAME} — {brain.model.num_params()/1e6:.1f}M params, {len(brain.kb.entries)} facts. /help for commands, Ctrl+C to quit.")
        try:
            while True:
                msg = input("you> ")
                r = brain.chat(msg)
                print("titan>", r["reply"])
                for n in r.get("notices", []):
                    print("     ", n)
        except (KeyboardInterrupt, EOFError):
            print()


if __name__ == "__main__":
    main()
