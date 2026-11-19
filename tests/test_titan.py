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
        self.assertEqual(kb.word_validity("I am Titan, a small free AI."), 1.0)
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
        self.assertIn(json.dumps(titan.UI_HTML), ui_js, "run python tools/build_worker_ui.py")


if __name__ == "__main__":
    unittest.main()
