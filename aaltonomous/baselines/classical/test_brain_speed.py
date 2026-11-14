"""brain speed knobs: image downscale for find/decide, find n_predict, and the finals_r22_agent defaults are wired.
Resizing must keep the find point mapping in ORIGINAL pixel coordinates (boxes are normalised 0-1000)."""
from __future__ import annotations

import base64
import http.server
import json
import threading
import unittest
from unittest import mock

try:
    import cv2
    import numpy as np
except ImportError:      # pragma: no cover
    cv2 = np = None

from aaltonomous.baselines.classical.brain import find as brain_find
from aaltonomous.baselines.classical.brain.find import FindClient, downscale_rgb
from aaltonomous.baselines.classical.config import load_config


class _Chat(http.server.BaseHTTPRequestHandler):
    bodies = []
    box = [500, 500, 600, 700]            # normalised 0-1000 -> centre (550, 600) per mille

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).bodies.append(body)
        out = json.dumps({"choices": [{"message": {"content": json.dumps([{"bbox_2d": self.box, "label": "x"}])}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def _server():
    _Chat.bodies = []
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Chat)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1/chat/completions"


def _sent_size(body):
    img = next(p for p in body["messages"][-1]["content"] if p["type"] == "image_url")["image_url"]["url"]
    arr = cv2.imdecode(np.frombuffer(base64.b64decode(img.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    return arr.shape[1], arr.shape[0]


@unittest.skipIf(cv2 is None, "needs cv2 + numpy")
class TestDownscale(unittest.TestCase):
    def test_downscale_keeps_aspect_and_reports_original(self):
        rgb = np.zeros((480, 640, 3), np.uint8)
        small, osz = downscale_rgb(rgb, 320)
        self.assertEqual((small.shape[1], small.shape[0]), (320, 240))
        self.assertEqual(osz, (640, 480))
        for m in (0, None, 640, 1000):                       # off / already small enough: untouched
            same, o = downscale_rgb(rgb, m)
            self.assertEqual(same.shape, rgb.shape)
            self.assertEqual(o, (640, 480))

    def test_find_points_are_in_original_pixels_after_resize(self):
        srv, url = _server()
        try:
            rgb = np.zeros((480, 640, 3), np.uint8)
            full = FindClient(path=None, url=url).detect(rgb, "the cube")
            for side in (384, 448, 512):
                _Chat.bodies.clear()
                got = FindClient(path=None, url=url, max_side=side).detect(rgb, "the cube")
                self.assertEqual(_sent_size(_Chat.bodies[0]), (side, round(480 * side / 640)))   # really downscaled
                self.assertEqual(got[0]["point"], full[0]["point"])                              # same original-pixel point
                self.assertEqual(got[0]["bbox"], full[0]["bbox"])
            self.assertEqual(full[0]["point"], (352.0, 288.0))                                  # 550/1000*640, 600/1000*480
            self.assertEqual(full[0]["bbox"], [320.0, 240.0, 384.0, 336.0])
        finally:
            srv.shutdown()

    def test_n_predict_cache_prompt_and_layout_reach_the_request(self):
        srv, url = _server()
        try:
            rgb = np.zeros((480, 640, 3), np.uint8)
            FindClient(path=None, url=url).detect(rgb, "the pan")                       # defaults = old behaviour
            d = _Chat.bodies[-1]
            self.assertEqual(d["max_tokens"], 400)
            self.assertTrue(d["cache_prompt"])
            self.assertEqual(d["messages"][0]["role"], "user")
            self.assertEqual(d["messages"][0]["content"][0]["type"], "image_url")      # image first, as before
            FindClient(path=None, url=url, n_predict=96, layout="static_first").detect(rgb, "the pan")
            d = _Chat.bodies[-1]
            self.assertEqual(d["max_tokens"], 96)
            self.assertEqual([m["role"] for m in d["messages"]], ["system", "user"])    # static prefix first, image + phrase last
            self.assertEqual(d["messages"][1]["content"][0]["type"], "image_url")
            self.assertIn("the pan", d["messages"][1]["content"][1]["text"])
            self.assertNotIn("the pan", d["messages"][0]["content"])
        finally:
            srv.shutdown()

    def test_old_find_tool_gets_no_new_kwargs_by_default(self):
        c = FindClient(path=None, url=None)
        self.assertEqual(c._kw(np.zeros((480, 640, 3)), (640, 480)), {"timeout": 1.5})


class TestWiring(unittest.TestCase):
    def test_config_defaults_are_the_old_behaviour(self):
        b = load_config()["brain"]
        self.assertEqual((b["image_max_side_px"], b["find_n_predict"], b["find_layout"]), (0, 400, "image_first"))

    def test_finals_r22_agent_has_the_speed_params(self):
        b = load_config(overlay="finals_r22_agent")["brain"]
        self.assertEqual(b["image_max_side_px"], 448)               # downscaled (never full) for the finals
        self.assertEqual(b["find_n_predict"], 128)
        self.assertLessEqual(b["agent_n_predict"], 160)

    @unittest.skipIf(cv2 is None, "needs cv2 + numpy")
    def test_policy_passes_params_to_find_client_and_resizes_decide_image(self):
        from aaltonomous.baselines.classical.mock_world import make_mock_env
        from aaltonomous.baselines.classical.policy import ClassicalPolicy
        pol = ClassicalPolicy(log=None, safety=make_mock_env(seed=0).filter.cfg, time_limit=120.0,
                              overrides={"brain.enabled": True, "brain.find": True, "brain.image_max_side_px": 448,
                                         "brain.find_n_predict": 96, "brain.find_layout": "static_first"})
        with mock.patch.object(brain_find, "FindClient") as fc, \
                mock.patch("aaltonomous.baselines.classical.policy.FindClient", fc):
            fc.return_value.return_value = []
            pol._brain_find_fn = None
            pol._brain_find("the pan", {"head_rgb": np.zeros((480, 640, 3), np.uint8)}, [])
        args = fc.call_args[0]
        self.assertEqual(args[3:], (448, 96, "static_first"))
        img = pol._brain_image({"head_rgb": np.zeros((480, 640, 3), np.uint8)})
        arr = cv2.imdecode(np.frombuffer(base64.b64decode(img), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual((arr.shape[1], arr.shape[0]), (448, 336))


if __name__ == "__main__":
    unittest.main()
