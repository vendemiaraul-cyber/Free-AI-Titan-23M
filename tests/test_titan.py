"""Tests for titan.py (tiny model, no network). Run: python -m unittest discover tests"""
import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import titan  # noqa: E402

TINY = titan.ModelConfig(n_layer=1, n_head=2, d_model=32, ctx=1536)


class FakeTeacher:
    mode = "worker"

    def __init__(self):
        self.calls = []

    def teach(self, question, context, draft, confidence, reason):
        self.calls.append((question, reason))
        return {"answer": "Lima is the capital of Peru.", "model": "@cf/openai/gpt-oss-120b",
                "report_id": "r1", "via": "worker"}


def make_brain(home: Path) -> titan.Brain:
    shutil.copy(ROOT / "TITAN.md", home / "TITAN.md")
    cfg = titan.Config(home=home, model=TINY, auto_train=False, learn_steps=2, batch_size=2, web_fallback=False,
                       perchance_url="")  # keep tests offline
    b = titan.Brain(cfg)
    b.teacher = FakeTeacher()
    return b


def wait_trainer(b, timeout=60):
    t0 = time.time()
    while (b.trainer.busy or b.trainer.q.qsize()) and time.time() - t0 < timeout:
        time.sleep(0.05)


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_seed_file_parses(self):
        kb = titan.KnowledgeBase(ROOT / "TITAN.md")
        self.assertGreaterEqual(len(kb.entries), 20)
        self.assertEqual(kb.search("Hello")[0][1].question, "Hello")
        self.assertEqual(kb.search("what is the capital of peru"), [])

    def test_word_validity_flags_garble(self):
        kb = titan.KnowledgeBase(ROOT / "TITAN.md")
        self.assertEqual(kb.word_validity("I am Titan, a free AI."), 1.0)
        self.assertLess(kb.word_validity("I am Tit I an flera fram questit"), 0.97)

    def test_add_and_replace(self):
        kb = titan.KnowledgeBase(self.dir / "TITAN.md")
        e = kb.add("What is 2+2?", "4", "user")
        self.assertEqual(e.answer, "4")
        kb.add("what is 2+2", "Four.", "cloud:test")
        self.assertEqual(len(kb.entries), 1)
        self.assertEqual(kb.entries[0].answer, "Four.")
        self.assertEqual(kb.entries[0].meta["source"], "cloud:test")
        # answers can't break the file structure
        kb.add("Tricky", "line\n### Q: injected\n<!-- x -->", "user")
        self.assertEqual(len(kb.entries), 2)


class ModelTests(unittest.TestCase):
    def test_default_size_is_23m(self):
        n = titan.Titan(titan.ModelConfig()).num_params()
        self.assertTrue(22_500_000 < n < 23_500_000, n)

    def test_kv_cache_matches_full_forward(self):
        import torch
        m = titan.Titan(titan.ModelConfig(n_layer=2, n_head=2, d_model=32, ctx=64)).eval()
        x = torch.randint(0, 256, (1, 16))
        full, _ = m(x)
        caches = [{} for _ in m.blocks]
        m(x[:, :10], caches=caches)
        step, _ = m(x[:, 10:11], caches=caches, pos_offset=10)
        rest, _ = m(x[:, 11:], caches=caches, pos_offset=11)
        self.assertLess((full[0, 10] - step[0, 0]).abs().max().item(), 1e-4)
        self.assertLess((full[0, 11:] - rest[0]).abs().max().item(), 1e-4)

    def test_training_reduces_loss(self):
        kb = titan.KnowledgeBase(ROOT / "TITAN.md")
        m = titan.Titan(titan.ModelConfig(n_layer=2, n_head=4, d_model=64, ctx=1536))
        losses = titan.train_steps(m, kb.entries[:4], 40, 2e-3, 4)
        self.assertLess(sum(losses[-5:]) / 5, sum(losses[:5]) / 5)

    def test_roundtrip_bytes(self):
        s = "héllo ✓ 世界"
        self.assertEqual(titan.decode(titan.encode(s)), s)


class BrainTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.b = make_brain(self.dir)

    def tearDown(self):
        wait_trainer(self.b)
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_unknown_question_triggers_cloud_call_and_md_update(self):
        r = self.b.chat("What is the capital of Peru?")
        self.assertTrue(r["learned"])
        self.assertEqual(r["reply"], "Lima is the capital of Peru.")
        self.assertIn("Automatic report", r["notices"][0])
        self.assertIn("gpt-oss-120b", r["notices"][0])
        md = (self.dir / "TITAN.md").read_text()
        self.assertIn("### Q: What is the capital of Peru?\nA: Lima is the capital of Peru.", md)
        self.assertIn("source: cloud:@cf/openai/gpt-oss-120b", md)
        rep = self.b.reports()[-1]
        self.assertEqual(rep["status"], "learned")
        wait_trainer(self.b)
        self.assertTrue((self.dir / "titan.pt").exists())
        self.assertEqual(self.b.trainer.history[-1]["kind"], "learn")

    def test_known_question_untrained_model_recalls_from_md(self):
        r = self.b.chat("Who is your teacher?")  # untrained model can't elaborate, but TITAN.md has it
        self.assertEqual(r["source"], "knowledge")
        self.assertIn("gpt-oss-120b", r["reply"])
        self.assertEqual(self.b.teacher.calls, [])

    def test_offline_queue(self):
        class Down:
            mode = "none"

            def teach(self, *a):
                raise titan.TeacherUnavailable("offline")
        self.b.teacher = Down()
        r = self.b.chat("What is the capital of Peru?")
        self.assertFalse(r["learned"])
        self.assertEqual(self.b.reports()[-1]["status"], "pending")
        self.b.teacher = FakeTeacher()
        done = self.b.process_pending()
        self.assertEqual(len(done), 1)
        self.assertIsNotNone(self.b.kb.find_exact("What is the capital of Peru?"))

    def test_commands(self):
        self.assertIn("/image", self.b.chat("/help")["reply"])
        r = self.b.chat("/teach What's my cat called? | Miso")
        self.assertIn("entry", r["reply"])
        self.assertEqual(self.b.kb.find_exact("What's my cat called?").answer, "Miso")
        r = self.b.chat("/image a cat")
        self.assertEqual(r["image"]["mode"], "unconfigured")
        self.assertEqual(titan.Config().perchance_url, "https://perchance.org/rg4m14ri")

    def test_animate_video(self):
        r = self.b.videos.generate("a red balloon", backend="animate", frames=6)
        self.assertTrue(Path(r["file"]).exists())
        self.assertEqual(r["frames"], 6)
        names = [x["name"] for x in self.b.videos.backends()]
        self.assertEqual(names[0], "animate")
        self.assertIn("cogvideox", names)

    def test_mcp(self):
        init = titan.mcp_handle(self.b, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                         "params": {"protocolVersion": "2025-06-18"}})
        self.assertEqual(init["result"]["serverInfo"]["name"], titan.NAME)
        self.assertIsNone(titan.mcp_handle(self.b, {"jsonrpc": "2.0", "method": "notifications/initialized"}))
        tools = titan.mcp_handle(self.b, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
        self.assertIn("titan_generate_video", [t["name"] for t in tools])
        r = titan.mcp_handle(self.b, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                      "params": {"name": "titan_knowledge", "arguments": {"query": "teacher"}}})
        self.assertIn("gpt-oss-120b", r["result"]["content"][0]["text"])


class SizeTests(unittest.TestCase):
    def test_presets_match_their_names(self):
        import torch
        expect = {"23m": (22.5e6, 23.5e6), "125m": (120e6, 130e6), "350m": (340e6, 360e6), "1.3b": (1.25e9, 1.35e9)}
        for name, (lo, hi) in expect.items():
            with torch.device("meta"):
                n = titan.Titan(titan.ModelConfig.preset(name)).num_params()
            self.assertTrue(lo < n < hi, (name, n))

    def test_resolve_and_checkpoint_names(self):
        self.assertEqual(titan.resolve_size("350m"), "350m")
        self.assertIn(titan.resolve_size("auto"), titan.MODEL_PRESETS)
        with self.assertRaises(SystemExit):
            titan.resolve_size("50b")
        self.assertEqual(titan.Config(home=Path("/x"), size="23m").ckpt_path.name, "titan.pt")
        self.assertEqual(titan.Config(home=Path("/x"), size="1.3b").ckpt_path.name, "titan-1.3b.pt")


# Real file listing of ali-vilab/text-to-video-ms-1.7b on the Hugging Face Hub (sizes in bytes).
MODELSCOPE_FILES = [
    ("model_index.json", 384), ("README.md", 6862), ("scheduler/scheduler_config.json", 465),
    ("text_encoder/config.json", 644), ("text_encoder/model.fp16.safetensors", 680821102),
    ("text_encoder/model.safetensors", 1361597016), ("text_encoder/pytorch_model.bin", 1361674657),
    ("text_encoder/pytorch_model.fp16.bin", 680900852), ("tokenizer/merges.txt", 524619),
    ("tokenizer/vocab.json", 1059962), ("unet/config.json", 787), ("unet/diffusion_pytorch_model.bin", 5645561389),
    ("unet/diffusion_pytorch_model.fp16.bin", 2823097059), ("unet/diffusion_pytorch_model.fp16.safetensors", 2822650226),
    ("unet/diffusion_pytorch_model.safetensors", 5645118394), ("vae/config.json", 657),
    ("vae/diffusion_pytorch_model.fp16.safetensors", 167335350), ("vae/diffusion_pytorch_model.safetensors", 334643276),
]
COMPONENTS = {"scheduler", "text_encoder", "tokenizer", "unet", "vae"}


class FakeDownloader(titan.VideoDownloader):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.fetched = []

    def _list_files(self, repo):
        return MODELSCOPE_FILES + [("assets/demo.png", 5000), ("big-single-file.safetensors", 9e9)]

    def _components(self, repo):
        return COMPONENTS

    def _fetch(self, repo, path):
        self.fetched.append(path)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_selects_only_what_diffusers_loads(self):
        fp16 = titan.select_video_files(MODELSCOPE_FILES, COMPONENTS, "fp16")
        paths = [p for p, _ in fp16]
        self.assertIn("unet/diffusion_pytorch_model.fp16.safetensors", paths)
        self.assertNotIn("unet/diffusion_pytorch_model.safetensors", paths)
        self.assertFalse(any(p.endswith(".bin") for p in paths))
        self.assertNotIn("README.md", paths)
        self.assertAlmostEqual(sum(s for _, s in fp16) / 1e9, 3.67, places=2)
        full = [p for p, _ in titan.select_video_files(MODELSCOPE_FILES, COMPONENTS, None)]
        self.assertIn("unet/diffusion_pytorch_model.safetensors", full)
        self.assertFalse(any(".fp16." in p for p in full))

    def test_download_progress_and_marker(self):
        cfg = titan.Config(home=self.dir, model=TINY)
        dl = FakeDownloader(cfg)
        self.assertEqual(dl.status("modelscope")["state"], "not downloaded")
        self.assertEqual(dl.status("animate")["state"], "built-in")
        seen = []
        dl.download("modelscope", lambda path, i, n, done: seen.append((i, n)))
        self.assertNotIn("assets/demo.png", dl.fetched)
        self.assertNotIn("big-single-file.safetensors", dl.fetched)
        self.assertEqual(seen[-1][0], seen[-1][1])
        st = dl.status("modelscope")
        self.assertEqual(st["state"], "downloaded")
        self.assertAlmostEqual(st["bytes_total"] / 1e9, 3.67, places=2)
        # a fresh downloader (e.g. after a restart) remembers it
        self.assertEqual(FakeDownloader(cfg).status("modelscope")["state"], "downloaded")
        with self.assertRaises(ValueError):
            dl.start("animate")

    def test_background_download_via_api(self):
        b = make_brain(self.dir)
        b.videos.downloads = FakeDownloader(b.cfg)
        srv = titan.make_server(b, "127.0.0.1", 0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            req = urllib.request.Request(base + "/v1/video/download", data=b'{"backend": "svd"}',
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as r:
                self.assertIn(json.loads(r.read())["download"]["state"], ("downloading", "downloaded"))
            for _ in range(100):
                with urllib.request.urlopen(base + "/v1/video/backends") as r:
                    svd = next(x for x in json.loads(r.read())["backends"] if x["name"] == "svd")
                if svd["download"]["state"] == "downloaded":
                    break
                time.sleep(0.05)
            self.assertEqual(svd["download"]["state"], "downloaded")
        finally:
            srv.shutdown()
            wait_trainer(b)


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        cls.b = make_brain(cls.dir)
        cls.srv = titan.make_server(cls.b, "127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        wait_trainer(cls.b)
        shutil.rmtree(cls.dir, ignore_errors=True)

    def get(self, path, raw=False):
        with urllib.request.urlopen(self.base + path) as r:
            data = r.read()
        return data if raw else data.decode()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    def test_routes(self):
        self.assertIn("Titan-23M", self.get("/"))
        self.assertIn('src="https://perchance.org/rg4m14ri"', self.get("/"))
        h = json.loads(self.get("/v1/health"))
        self.assertEqual(h["teacher"], "gpt-oss-120b")
        self.assertIn("## Knowledge", self.get("/v1/knowledge"))
        r = self.post("/v1/chat", {"message": "What is the capital of Peru?"})
        self.assertTrue(r["learned"])
        oa = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Who is your teacher?"}]})
        self.assertIn("gpt-oss-120b", oa["choices"][0]["message"]["content"])
        v = self.post("/v1/video", {"prompt": "sunrise", "backend": "animate", "frames": 4})
        self.assertTrue(v["video_url"].startswith("/outputs/"))
        self.assertGreater(len(self.get(v["video_url"], raw=True)), 100)
        t = self.post("/v1/teach", {"question": "Favourite colour?", "answer": "Blue"})
        self.assertTrue(t["ok"])
        m = self.post("/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertTrue(m["result"]["tools"])
        self.assertTrue(json.loads(self.get("/v1/video/backends"))["backends"])

    def test_worker_ui_in_sync(self):
        ui_js = (ROOT / "cloudflare" / "src" / "ui.js").read_text()
        self.assertIn(json.dumps(titan.UI_HTML, ensure_ascii=False), ui_js, "run python tools/build_worker_ui.py")
        seed_js = (ROOT / "cloudflare" / "src" / "seed.js").read_text()
        self.assertIn(json.dumps((ROOT / "TITAN.md").read_text(), ensure_ascii=False), seed_js,
                      "run python tools/build_worker_ui.py")


if __name__ == "__main__":
    unittest.main()
