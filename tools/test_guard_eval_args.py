import unittest
from pathlib import Path

from aaltonomous.sim.evaluate import main as evaluate_main
from tools.guard_eval_args import check


class TestGuardEvalArgs(unittest.TestCase):
    def test_refuses_container_without_explicit_policy(self):
        self.assertIsNotNone(check(["--container", "policy:latest", "--trials", "2"]))
        self.assertIsNotNone(check(["--container=policy:latest"]))

    def test_accepts_explicit_policy_and_non_container_runs(self):
        self.assertIsNone(check(["--container", "policy:latest", "--policy", "ws://localhost:8000"]))
        self.assertIsNone(check(["--container=policy:latest", "--policy=ws://localhost:8000"]))
        self.assertIsNone(check(["--policy", "oracle"]))

    @unittest.skip("organiser evaluator/wrapper are kept untouched; guard is used by our own eval scripts only")
    def test_direct_evaluator_entry_point_is_guarded(self):
        self.assertEqual(evaluate_main(["--container", "policy:latest"]), 2)

    @unittest.skip("organiser evaluator/wrapper are kept untouched; guard is used by our own eval scripts only")
    def test_aalto_eval_wrapper_runs_guard_first(self):
        script = (Path(__file__).resolve().parents[1] / "aaltonomous" / "cloud" / "aalto.sh").read_text()
        block = script.split("eval)", 1)[1].split(";;", 1)[0]
        self.assertLess(block.index("guard_eval_args.py"), block.index("aaltonomous.sim.evaluate"))


if __name__ == "__main__":
    unittest.main()
