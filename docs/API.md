# Titan-23M API

Both editions speak the same API. `titan.py serve` listens on `http://127.0.0.1:8023`; the
Cloudflare edition is at your Worker URL. Requests and responses are JSON unless noted.

**Auth.** `titan.py`: if `TITAN_LOCAL_KEY` is set, send `Authorization: Bearer <key>` (everything
except `/`, `/outputs/*` and `/v1/health`). Worker: if the `TITAN_API_KEY` secret is set, the same
header is required on everything except `/`, `/v1/health`, `GET /v1/knowledge`,
`/v1/video/backends`, `/v1/image/status`, and `POST /v1/chat` while `PUBLIC_CHAT="true"`.

| | titan.py | Worker |
|---|---|---|
| `GET /` browser UI | ✅ | ✅ |
| `GET /v1/health` | ✅ | ✅ |
| `GET /v1/model` | ✅ | – |
| `POST /v1/chat` | ✅ PyTorch model + TITAN.md + teacher | ✅ TITAN.md + teacher |
| `POST /v1/chat/completions` | ✅ | ✅ |
| `POST /v1/teach` | ✅ (needs `answer`) | ✅ (asks gpt-oss-120b if no `answer`) |
| `POST /v1/learn` | ✅ | ✅ |
| `GET/POST /v1/knowledge`, `GET /v1/knowledge/entries` | ✅ | ✅ |
| `POST /v1/image`, `GET /v1/image/status` | ✅ headless Perchance | ✅ returns your Perchance link |
| `POST /v1/video`, `GET /v1/video/backends` | ✅ | 501 (use browser `animate`) |
| `GET/POST /v1/video/download` | ✅ | – |
| `GET /v1/search?q=` | ✅ | ✅ |
| `GET /v1/reports` | ✅ | ✅ |
| `POST /v1/train` | ✅ | – |
| `POST /mcp` | ✅ | ✅ |

---

## Chat

### `POST /v1/chat`

```json
{ "message": "What is the capital of Peru?" }
```

```json
{
  "reply": "Lima is the capital of Peru.",
  "source": "cloud:@cf/openai/gpt-oss-120b",
  "confidence": 0.12,
  "learned": true,
  "entry": 27,
  "notices": ["🔔 Automatic report: Titan couldn't elaborate a response (confidence 0.12, nothing about it in TITAN.md). Cloud call → learned from gpt-oss-120b on Cloudflare → TITAN.md updated (entry #27) → training on it in the background."]
}
```

`source` is one of `titan` (the model answered by itself), `knowledge` (recalled from TITAN.md),
`cloud:<model>`, `web:<site>`, `none` (queued while offline), or a command name.

Slash commands work in `message`: `/help`, `/image <prompt>`, `/video[:backend] <prompt>`,
`/search <q>`, `/teach <q> | <a>`, `/learn <q>`, `/tools`, `/tool <name> <json>`, `/knowledge`,
`/status`. Image and video commands add `image` / `video` objects with `image_url` / `video_url`.

### `POST /v1/chat/completions` (OpenAI-compatible)

```bash
curl http://127.0.0.1:8023/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"titan-23m","messages":[{"role":"user","content":"Who are you?"}]}'
```

Returns a standard `chat.completion` object plus a `titan` field holding the notices, source
and confidence. Works with OpenAI SDKs (`base_url="http://127.0.0.1:8023/v1"`). Only the last user
message is used; streaming isn't supported.

## Learning and knowledge

### `POST /v1/teach`

* `{ "question": "...", "answer": "..." }` saves your answer to TITAN.md (and `titan.py` trains on it).
* Worker only: `{ "question": "...", "context": "...", "draft": "...", "confidence": 0.2, "reason": "...", "client": "titan.py/1.0.0" }`
  is the **cloud call**. gpt-oss-120b writes the answer, the Worker appends it to its TITAN.md,
  stores a report (and posts it to `REPORT_WEBHOOK_URL`), and returns
  `{ "answer", "model", "report_id", "entry", "message" }`.

### `POST /v1/learn`

`{ "question": "..." }` forces a cloud lesson even if Titan thinks it knows the answer.

### `GET /v1/knowledge`

Returns TITAN.md as `text/markdown`. `GET /v1/knowledge/entries?q=...` returns parsed entries
(`id`, `question`, `answer`, `meta`), searched when `q` is given.

### `POST /v1/knowledge`

`{ "markdown": "<another TITAN.md>" }` merges entries you don't have yet and returns
`{ "added", "entries" }`. `titan.py sync` uses this.

### `GET /v1/reports`

The automatic reports, newest last (`titan.py`) or newest first (Worker): question, reason,
confidence, status (`learned` / `pending`), teacher, entry.

### `POST /v1/train` (titan.py)

`{ "steps": 300 }` queues a full training run on TITAN.md in the background. `GET /v1/health` and `GET /v1/model`
report the running `size` (`23m`, `125m`, `350m` or `1.3b`), its `parameters` and `config`.

## Images (Perchance)

### `POST /v1/image`

```json
{ "prompt": "a lighthouse on a cliff at sunset, oil painting", "negative_prompt": "blurry" }
```

titan.py with Playwright:
`{ "ok": true, "mode": "headless", "file": ".../outputs/image_....png", "image_url": "/outputs/image_....png" }`.

Without Playwright, or on the Worker:
`{ "ok": false, "mode": "browser", "url": "https://perchance.org/rg4m14ri", "message": "..." }`.
The browser UI embeds that generator in the Image tab and copies the prompt to your clipboard.

`GET /v1/image/status` shows whether a generator is configured and headless mode is available.

## Video (on your device)

### `GET /v1/video/backends`

Lists `animate`, `cogvideox`, `ltx`, `wan`, `modelscope` and `svd`, each with `kind`, `online`,
`available`, `missing` (packages), `where` and `warning` (e.g. no GPU).

Each heavy backend also has `download`, the state of its weights on this device: `not downloaded`,
`downloading` (with `bytes_done`, `bytes_total`, `files_done`, `files_total`, `current`), `downloaded`
or `error` (with `error`). When its packages are missing, `pip` holds the command that installs them.

### `POST /v1/video/download` (titan.py)

`{ "backend": "ltx" }` starts downloading that generator's weights in the background and returns its
`download` state. Poll `GET /v1/video/download` (all backends) or `GET /v1/video/backends` for
progress. Only the files the diffusers pipeline loads are downloaded. Gated models (SVD) need
`HF_TOKEN` in the environment.

### `POST /v1/video`

```json
{ "prompt": "a paper boat drifting on a rainy street", "backend": "ltx",
  "frames": 97, "fps": 24, "steps": 40, "seed": 7, "negative_prompt": "",
  "image_base64": null }
```

Returns `{ "ok": true, "backend", "file", "video_url", "frames", "fps", "seed", "notes" }`. The
output is MP4 when `imageio-ffmpeg` is installed, otherwise an animated GIF. Image-to-video
backends (`animate`, `svd`) use `image_base64`, or generate a first frame with Perchance.

## Web search connector

`GET /v1/search?q=alan+turing` returns `{ "query", "results": [{ "title", "url", "snippet", "source" }] }`
from DuckDuckGo instant answers and Wikipedia.

## MCP

`POST /mcp` takes JSON-RPC 2.0 (single or batch) in the Streamable-HTTP style, with JSON
responses. `python titan.py mcp` is the same server over stdio.

| Tool | titan.py | Worker |
|---|---|---|
| `titan_chat` `{message}` | ✅ | ✅ |
| `titan_generate_image` `{prompt, negative_prompt?}` | ✅ returns the image | ✅ returns the link |
| `titan_generate_video` `{prompt, backend?, frames?, image_base64?}` | ✅ | – |
| `titan_download_video_generator` `{backend}` | ✅ | – |
| `titan_web_search` `{query}` | ✅ | ✅ |
| `titan_teach` | `{question, answer}` | `{question}` → gpt-oss-120b |
| `titan_knowledge` `{query?}` | ✅ | ✅ |

```bash
curl http://127.0.0.1:8023/mcp -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"titan_chat","arguments":{"message":"hello"}}}'
```
