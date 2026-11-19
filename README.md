# Free-AI-Titan-23M

The name doesn't give any specs, but it's both downloadable or directly talkable in the browser,
you can also connect it to an MCP server or ask it a connector (web search).

Titan is a small AI (**~23M parameters, written in PyTorch**) that keeps its memory in a
Markdown file, **`TITAN.md`**. When it can't come up with an answer, it calls its teacher,
**gpt-oss-120b on Cloudflare Workers AI**, posts an automatic message to say so, writes the
lesson into `TITAN.md` and trains itself on it.

It comes in two editions:

| | **Single-file edition** — `titan.py` | **Cloudflare edition** — `cloudflare/` |
|---|---|---|
| Where | Downloaded, runs on your device | Your Cloudflare Worker, talk to it in the browser |
| Brain | The PyTorch Titan-23M model + `TITAN.md` | `TITAN.md` in Workers KV + gpt-oss-120b |
| Images | Your Perchance generator, driven automatically | Your Perchance generator (opens it) |
| Video | All generators, on your device | Only `animate`, in your browser |
| API | REST, OpenAI-compatible chat, MCP (stdio + HTTP) | REST, OpenAI-compatible chat, MCP (HTTP) |

## How Titan learns

```
you ──► Titan-23M (PyTorch) ──► confident? ──yes──► answer
              ▲  reads the closest            │
              │  TITAN.md entry               no (low confidence / nothing in TITAN.md)
              │                               ▼
              │                  cloud call: POST <worker>/v1/teach
              │                  gpt-oss-120b on Cloudflare writes the answer
              │                               │
              │      🔔 automatic report message (chat notice, titan_reports.jsonl,
              │         KV report on Cloudflare, optional Discord/Slack webhook)
              │                               │
              └── fine-tunes in the background ◄── answer appended to TITAN.md
```

* **Confidence** is the geometric-mean probability of the tokens Titan generated, multiplied by
  the cube of the share of its words that exist in `TITAN.md`. (A byte-level model can be "sure"
  of a garbled word, so real words are required too.) Titan counts as unable to elaborate when
  confidence is below `TITAN_CONF_THRESHOLD` (0.75), when under 97% of its words are real, or
  when nothing relevant is in `TITAN.md` and it isn't very confident (0.90).
* If the answer is already in `TITAN.md` but the model hasn't learned it yet, Titan answers from
  the file and re-trains on that entry, with no cloud call.
* If the cloud is unreachable, Titan falls back to Wikipedia (`TITAN_WEB_FALLBACK=1`), or queues
  the question in `titan_reports.jsonl`. `python titan.py sync` learns queued questions later.
* Training runs on a copy of the model in a background thread, then swaps the new weights in, so
  chatting never waits for it. Weights are saved to `titan.pt`.

## Single-file edition (`titan.py`)

```bash
pip install torch pillow
python titan.py train            # first training on TITAN.md (≈2.5 s/step on a laptop CPU, seconds on a GPU)
python titan.py serve            # browser UI + API on http://127.0.0.1:8023
python titan.py chat             # terminal chat
python titan.py ask "Who are you?"
python titan.py image "a lighthouse at sunset"
python titan.py video "a paper boat on a rainy street" --backend ltx
python titan.py mcp              # MCP server over stdio
python titan.py sync             # pull/push TITAN.md with the cloud + learn queued questions
python titan.py info
```

If there's no `titan.pt` yet, `serve` and `chat` start a background bootstrap training
automatically.

### Images: Titan's Perchance generator

Titan's image generator is **https://perchance.org/rg4m14ri**. It's the default, and the Image
tab of the browser UI embeds it. For automatic images through the API and `/image`:

```bash
pip install playwright && playwright install chromium
# export TITAN_PERCHANCE_URL=https://perchance.org/...   # only to use a different generator
```

Titan opens the generator headlessly, types the prompt, clicks generate and saves the image to
`outputs/`. See [`perchance/README.md`](perchance/README.md) for the element IDs it looks for and
an optional starter template. Without Playwright you get the generator link instead.

### Video generators (on your device)

| Backend | Type | Model | Online? |
|---|---|---|---|
| `animate` | image → video | Ken Burns pan/zoom (CPU, no download) | ✅ also runs in the browser |
| `cogvideox` | text → video | `THUDM/CogVideoX-2b` | device only |
| `ltx` | text → video | `Lightricks/LTX-Video` | device only |
| `wan` | text → video | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | device only |
| `modelscope` | text → video | `ali-vilab/text-to-video-ms-1.7b` | device only |
| `svd` | image → video | `stabilityai/stable-video-diffusion-img2vid-xt` | device only |

Heavy backends need `pip install diffusers transformers accelerate sentencepiece imageio imageio-ffmpeg`
and realistically an NVIDIA GPU. Weights download once on first use. Image-to-video backends
animate your uploaded image, or a fresh Perchance image when none is given.

### Connect to MCP servers and the web

```bash
export TITAN_MCP_SERVERS=https://my-mcp-server.example/mcp,https://other.example/mcp
```

In chat: `/tools` lists their tools, `/tool <name> {"arg": "value"}` calls one, and `/search <query>`
uses the built-in web connector (DuckDuckGo instant answers + Wikipedia, no key needed).

Use Titan itself as an MCP server, e.g. in Claude Desktop or Claude Code:

```json
{ "mcpServers": { "titan": { "command": "python", "args": ["/path/to/titan.py", "mcp"] } } }
```

or over HTTP at `http://127.0.0.1:8023/mcp` while `titan.py serve` runs.

## Cloudflare edition (`cloudflare/`)

The browser edition of Titan, running as a Cloudflare Worker. It shares the browser UI with
`titan.py`: after editing `UI_HTML`, run `python tools/build_worker_ui.py`.

## API

Full reference: [`docs/API.md`](docs/API.md). Highlights:

| Method & path | What |
|---|---|
| `POST /v1/chat` `{message}` | Chat (slash commands work too); returns `reply`, `confidence`, `source`, `notices` |
| `POST /v1/chat/completions` | OpenAI-compatible chat |
| `POST /v1/teach` `{question, answer?}` | Teach Titan; without `answer` the Worker asks gpt-oss-120b |
| `POST /v1/learn` `{question}` | Force a cloud lesson |
| `GET /v1/knowledge` · `POST /v1/knowledge` `{markdown}` | Read / merge `TITAN.md` |
| `POST /v1/image` `{prompt, negative_prompt}` | Perchance image |
| `POST /v1/video` `{prompt, backend, frames, image_base64}` · `GET /v1/video/backends` | Video |
| `GET /v1/search?q=` | Web search connector |
| `GET /v1/reports` | Automatic learning reports |
| `POST /mcp` | MCP (JSON-RPC) |

## Tests

```bash
python -m unittest discover tests     # titan.py (tiny model, no network)
node --test cloudflare/test/*.test.mjs # the Worker, with fake AI + KV
```

## Configuration (`titan.py`)

| Variable | Default | Meaning |
|---|---|---|
| `TITAN_HOME` | folder of `titan.py` | where `TITAN.md`, `titan.pt`, reports and `outputs/` live |
| `TITAN_TEACHER_MODEL` | `@cf/openai/gpt-oss-120b` | the teacher |
| `TITAN_WEB_FALLBACK` | `1` | learn from Wikipedia when the cloud is down |
| `TITAN_PERCHANCE_URL` (+ `_PROMPT/_NEGATIVE/_BUTTON_SELECTOR`) | `https://perchance.org/rg4m14ri` | the Perchance image generator |
| `TITAN_MCP_SERVERS` | – | comma-separated MCP server URLs to connect to |
| `TITAN_CONF_THRESHOLD`, `TITAN_SIM_THRESHOLD` | `0.75`, `0.30` | when Titan counts as unable to elaborate |
| `TITAN_LEARN_STEPS`, `TITAN_BOOTSTRAP_STEPS`, `TITAN_BATCH_SIZE` | `30`, `300`, `8` | training amounts |
| `TITAN_AUTO_TRAIN` | `1` | bootstrap-train when there's no checkpoint |
| `TITAN_LOCAL_KEY` | – | require `Authorization: Bearer <key>` on the local API |
| `TITAN_HOST`, `TITAN_PORT` | `127.0.0.1`, `8023` | server address |
