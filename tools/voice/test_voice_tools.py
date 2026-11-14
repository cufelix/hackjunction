from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

import yaml

from tools.voice.generate_clips import generate_all
from tools.voice.sidecar import ClipCatalog, EventProcessor, NullPlayer, clip_keys, parse_event


class TestSidecar(unittest.TestCase):
    def test_parse_and_event_mapping(self):
        event = parse_event(b'{"event":"placed","colour":"yellow","shape":"cube","bin":"left"}')
        self.assertEqual(clip_keys(event)[:2], ["placed.yellow.cube.left", "placed.left"])
        with self.assertRaises(ValueError):
            parse_event(b'{"event":"move_arm"}')

    def test_deduplicates_with_null_backend_and_done_has_priority(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "clips").mkdir()
            cfg = {"clips": {"episode_start": {"text": "Start"},
                              "episode_done.default": {"text": "Done"}}}
            (root / "clips.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
            for filename in ("start.wav", "done.wav"):
                (root / "clips" / filename).write_bytes(b"not decoded by null")
            manifest = {"clips": {"episode_start": {"file": "start.wav"},
                                  "episode_done.default": {"file": "done.wav"}}}
            (root / "clips" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            catalog = ClipCatalog(root / "clips.yaml")
            player = NullPlayer(log=None)
            now = [10.0]
            proc = EventProcessor(catalog, player, dedupe_s=3.0, min_interval_s=0,
                                  clock=lambda: now[0], sleeper=lambda _s: None, log=None)
            self.assertTrue(proc.submit({"event": "episode_start"}))
            self.assertFalse(proc.submit({"event": "episode_start"}))
            self.assertTrue(proc.submit({"event": "episode_done", "count": 1}))
            self.assertTrue(proc.play_one())
            self.assertEqual(player.played[0].name, "done.wav")
            self.assertTrue(proc.play_one())
            self.assertEqual(player.played[1].name, "start.wav")

    def test_bounded_queue_drops_without_waiting(self):
        class MissingCatalog:
            resolve = lambda *_args: (None, None)
        proc = EventProcessor(MissingCatalog(), NullPlayer(log=None), queue_size=1, log=None)
        self.assertTrue(proc.submit({"event": "picked", "colour": "red"}))
        self.assertFalse(proc.submit({"event": "picked", "colour": "blue"}))

    def test_done_replaces_lower_priority_event_when_queue_is_full(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "clips").mkdir()
            (root / "clips.yaml").write_text("clips: {}\n", encoding="utf-8")
            (root / "clips" / "done.wav").write_bytes(b"null backend does not decode")
            (root / "clips" / "manifest.json").write_text(json.dumps({"clips": {
                "episode_done.default": {"file": "done.wav"}}}), encoding="utf-8")
            player = NullPlayer(log=None)
            proc = EventProcessor(ClipCatalog(root / "clips.yaml"), player, queue_size=1,
                                  min_interval_s=0, log=None)
            self.assertTrue(proc.submit({"event": "picked", "colour": "red"}))
            self.assertTrue(proc.submit({"event": "episode_done", "count": 1}))
            self.assertTrue(proc.play_one())
            self.assertEqual(player.played[0].name, "done.wav")


class TestGenerateClips(unittest.TestCase):
    def test_dry_run_uses_no_http_and_mocked_generation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "clips.yaml"
            config.write_text(yaml.safe_dump({"clips": {
                "episode_start": {"text": "Sorting run started."},
                "rap_intro": {"text": "placeholder", "generate": False},
            }}), encoding="utf-8")
            out = root / "clips"
            calls = []

            def mock_http(url, headers, body):
                calls.append((url, headers, json.loads(body)))
                return b"\x00\x00" * 160

            generated, skipped, planned = generate_all(
                config, out, voice_id="voice-test", dry_run=True,
                api_key="unused", http_post=lambda *_args: self.fail("HTTP called during dry-run"), log=lambda *_: None)
            self.assertEqual((generated, skipped, planned), ([], [], ["episode_start"]))
            generated, skipped, planned = generate_all(
                config, out, voice_id="voice-test", api_key="test-key", http_post=mock_http, log=lambda *_: None)
            self.assertEqual(generated, ["episode_start"])
            self.assertEqual(len(calls), 1)
            self.assertNotIn("test-key", (out / "manifest.json").read_text(encoding="utf-8"))
            with wave.open(str(out / "episode_start.wav"), "rb") as wav:
                self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()), (16000, 1, 2))
            generated, skipped, planned = generate_all(
                config, out, voice_id="voice-test", api_key="test-key", http_post=mock_http, log=lambda *_: None)
            self.assertEqual((generated, skipped, planned), ([], ["episode_start"], []))
            self.assertEqual(len(calls), 1)

    def test_library_has_requested_size(self):
        config = Path(__file__).with_name("clips.yaml")
        entries = (yaml.safe_load(config.read_text(encoding="utf-8")) or {})["clips"]
        self.assertGreaterEqual(len(entries), 20)
        self.assertLessEqual(len(entries), 30)


if __name__ == "__main__":
    unittest.main()
