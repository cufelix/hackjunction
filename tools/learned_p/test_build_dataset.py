import tempfile
import unittest
from pathlib import Path

from tools.learned_p.build_dataset import parse_log


class TestBuildDataset(unittest.TestCase):
    def test_desired_bin_does_not_hide_actual_counter_drop(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "policy.log"
            log.write_text(
                "[learned_p] t=1.00 colour=red shape=cube hand=left actual_hand=left "
                "target_bin=right actual_target=counter attempt=1 x=0.4 y=0.1 distance=0.412 "
                "nn=na time_remaining=119 p_prior=0.8 p_final=0.8 duration_prior=5 duration_model=5\n"
                "[classical t= 1.0] execute red left arm -> counter bin "
                "grasp=(0.40000,0.10000,0.20000) 5.0s safety_clips=0\n"
                "[policy] verified red in right slot 0\n"
            )
            rows = parse_log(log)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["target_bin"], "right")
            self.assertEqual(rows[0]["label"], 0)
            self.assertEqual(rows[0]["outcome_source"], "non_target_drop")


if __name__ == "__main__":
    unittest.main()
