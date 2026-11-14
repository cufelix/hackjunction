"""python3 -m unittest tools.test_triage_real -v  (stdlib only)"""
import csv
import io
import os
import tempfile
import unittest

from tools import triage_real as tr

LOG = """\
[classical t=  1.00] held calibration: empty close at home
[retry_vary] orange label=prism try 1: x
[classical t=  5.00] execute orange left  arm -> left bin  grasp=(+0.31,-0.16,+0.05) drop=(+0.45,+0.19,+0.20)  IK ok max_err=2.0mm max_tilt=10deg  8.0s  safety_clips=0
[held_abort] t=7.0 orange left target=left close sample empty: carry cancelled, proving vertical escape
[classical t= 15.00] execute orange left  arm -> left bin  grasp=(+0.31,-0.16,+0.05) drop=(+0.45,+0.19,+0.20)  IK ok max_err=2.0mm max_tilt=10deg  8.0s  safety_clips=0
[grasp_diag] t=17.0 orange left target=left held=yes f=1
[landing] t=24.0 orange->left: lost delta={'left': 0.1}
[classical t= 25.00] execute purple right arm -> center bin  grasp=(+0.30,+0.10,+0.05) drop=(+0.50,+0.00,+0.20)  IK ok max_err=1.0mm max_tilt=5deg  7.0s  safety_clips=0
[grasp_diag] t=27.0 purple right target=center held=yes f=1
[landing] t=33.0 purple->center: ok delta={'center': 0.9}
[classical t= 34.00] execute yellow right arm -> right bin  grasp=(+0.30,+0.20,+0.05) drop=(+0.50,-0.20,+0.20)  IK ok max_err=1.0mm max_tilt=5deg  7.0s  safety_clips=0
[grasp_diag] t=36.0 yellow right target=right held=yes f=1
[landing] t=41.0 yellow->right: wrong_bin delta={'center': 0.8}
[landing] yellow landed in center (wrong bin, -1): logged, not corrected
[classical t= 42.00] execute yellow left  arm -> right bin  grasp=(+0.30,+0.20,+0.05) drop=(+0.50,-0.20,+0.20)  IK ok max_err=1.0mm max_tilt=5deg  7.0s  safety_clips=2
[classical t=120.00] episode time limit reached
[classical] episode end: attempts=5 failed=2 too_long=0 started=5 retry_pass_used=0
[classical t=  4.00] execute purple right arm -> center bin  grasp=(+0.30,+0.10,+0.05) drop=(+0.50,+0.00,+0.20)  IK ok max_err=1.0mm max_tilt=5deg  7.0s  safety_clips=0
[classical t= 11.00] verified purple in center slot 0
[classical t= 20.00] execute orange right arm -> left bin  grasp=(+0.30,+0.10,+0.05) drop=(+0.50,+0.00,+0.20)  IK ok max_err=1.0mm max_tilt=5deg  7.0s  safety_clips=0
"""


class TestTriage(unittest.TestCase):
    def setUp(self):
        self.picks = tr.parse(io.StringIO(LOG))

    def test_tags_in_order(self):
        self.assertEqual([p["tag"] for p in self.picks],
                         ["missed_grasp", "carry_drop", "ok", "wrong_bin", "clip", "ok", "unknown"])

    def test_episodes_and_pick_index(self):
        self.assertEqual([(p["episode"], p["pick"]) for p in self.picks],
                         [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (2, 2)])

    def test_fields(self):
        p = self.picks[0]
        self.assertEqual((p["color"], p["side"], p["target"], p["label"]), ("orange", "left", "left", "prism"))
        self.assertEqual(p["grasp"], (0.31, -0.16, 0.05))
        self.assertEqual(self.picks[4]["safety_clips"], 2)
        self.assertTrue(self.picks[4]["timeout"])

    def test_timeout_without_clip(self):
        log = LOG.replace("safety_clips=2", "safety_clips=0")
        self.assertEqual(tr.parse(io.StringIO(log))[4]["tag"], "timeout")

    def test_landing_for_other_colour_ignored(self):
        log = ("[classical t= 1.00] execute orange left  arm -> left bin  grasp=(+0.3,+0.1,+0.0) safety_clips=0\n"
               "[landing] t=2.0 purple->center: wrong_bin delta={}\n")
        self.assertEqual(tr.parse(io.StringIO(log))[0]["tag"], "unknown")

    def test_summary_grid(self):
        s = tr.summary(self.picks)
        self.assertEqual(s["by_side_object"]["left/prism"], {"missed_grasp": 1, "carry_drop": 1})
        self.assertEqual(s["n"], 7)

    def test_video_overrides_and_reports_unmatched(self):
        rows = [{"episode": "2", "pick": "2", "tag": "carry_drop", "note": "fell over the left rim"},
                {"episode": "9", "pick": "1", "tag": "ok", "note": ""}]
        un = tr.apply_video(self.picks, rows)
        self.assertEqual(self.picks[6]["tag"], "carry_drop")
        self.assertEqual(self.picks[6]["tag_source"], "video")
        self.assertEqual(len(un), 1)
        with self.assertRaises(ValueError):
            tr.apply_video(self.picks, [{"episode": "1", "pick": "1", "tag": "bogus"}])

    def test_cli_two_logs_offset_episodes(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, v, j, m = (os.path.join(d, x) for x in ("a.log", "b.log", "v.csv", "o.json", "o.md"))
            for path in (a, b):
                with open(path, "w") as f:
                    f.write(LOG)
            with open(v, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["episode", "pick", "tag", "note"])
                w.writerow(["3", "1", "missed_grasp", "x"])
            self.assertEqual(tr.main([a, b, "--video", v, "--json", j, "--md", m]), 0)
            import json
            out = json.load(open(j))
            self.assertEqual(max(p["episode"] for p in out["picks"]), 4)
            self.assertEqual(out["picks"][7]["tag"], "missed_grasp")
            self.assertIn("left/prism", open(m).read())


if __name__ == "__main__":
    unittest.main()
