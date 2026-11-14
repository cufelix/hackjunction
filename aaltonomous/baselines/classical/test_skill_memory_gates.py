"""Skill-memory gates (T7): validated-only selection, SQLite reload, fallback without raising, flag-off identity."""
import contextlib
import hashlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from .config import load_config
from .policy import ClassicalPolicy
from .skill_memory import ARMS, SkillMemory

KEY = ("cube", "red", "left")
DEFAULT = ["stock", "yaw_positive", "yaw_negative"]


@contextlib.contextmanager
def no_env_path():
    with mock.patch.dict(os.environ):
        os.environ.pop("AALTO_SKILLS_DB", None)
        yield


class Planner:
    """Stub planner: `infeasible` = set of variant names the planner reports as not ok."""

    def __init__(self, infeasible=(), target_for=None):
        self.vary = None
        self.infeasible = set(infeasible)
        self.target_for = target_for or {}

    def plan(self, det, arm, hand, **kw):
        v = dict(self.vary or {})
        name = self._name(v)
        yaw = 0.0 if v.get("yaw") is None else v["yaw"]
        plan = SimpleNamespace(ok=name not in self.infeasible, side="left",
                               posture=(v.get("posture") or "closed_tight", None, 1.0, "metric"),
                               waypoints={"grasp": det.top.copy()},
                               target_bin=self.target_for.get(name, "bin"), heading=yaw,
                               reason="infeasible (stub)")
        return plan

    @staticmethod
    def _name(v):
        """Name a variant the way policy._skill_arm does (prev yaw is 0 in these tests)."""
        if not v:
            return "stock"
        alt = v.get("posture") == "closed_demo"
        if v.get("yaw") is None:
            return "alternate_posture" if alt else "stock"
        return ("alternate_posture_yaw_" if alt else "yaw_") + ("positive" if v["yaw"] >= 0 else "negative")

    @staticmethod
    def finger_heading(plan):
        return plan.heading


def make_policy(memory, planner, retry=False):
    policy = ClassicalPolicy.__new__(ClassicalPolicy)
    policy.cfg = {"policy": {"retry_vary_grasp": retry, "retry_assoc_radius": 0.06,
                             "retry_yaw_deg": {"none": 90, "default": 90}, "retry_max_variants": 3},
                  "grasp": {"close_posture": "closed_tight"}}
    policy.skill_memory = memory
    policy.planner = planner
    policy.arm_cmd = policy.hand_cmd = np.zeros(14)
    policy.vary_hist = []
    policy.log = lambda _: None
    policy._decision_motion = lambda: contextlib.nullcontext(planner)
    return policy


def learn(memory, key, good, bad, n=6):
    for _ in range(n):
        memory.record(key, bad, "miss", 0.1)
        memory.record(key, good, "placed", 0.1)


class TestValidatedSelection(unittest.TestCase):
    """(a) skill memory can only pick among planner-validated variants of the existing retry-vary set."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_learned_but_infeasible_variant_is_never_planned_as_first_try(self):
        memory = SkillMemory(str(self.root / "a.sqlite"), seed=0, log=lambda _: None)
        learn(memory, ("unknown", "red", "left"), "alternate_posture", "stock")
        policy = make_policy(memory, Planner(infeasible={"alternate_posture"}))
        det = SimpleNamespace(color="red", shape=None, top=np.array([0.3, 0.1, 0.8]))
        plan = policy._plan_pick(det, {})
        self.assertTrue(plan.ok)
        self.assertEqual(plan.skill_arm, "stock")
        self.assertIsNone(getattr(policy.planner, "vary", None))
        memory.close()

    def test_retry_path_skips_infeasible_learned_arm_and_uses_next_validated_one(self):
        memory = SkillMemory(str(self.root / "b.sqlite"), seed=0, log=lambda _: None)
        learn(memory, KEY, "yaw_positive", "stock")
        planner = Planner(infeasible={"yaw_positive"})
        policy = make_policy(memory, planner, retry=True)
        policy.vary_hist = [{"color": "red", "xy": np.array([0.3, 0.1]), "yaw": 0.0,
                             "posture": "closed_tight", "side": "left", "held": False}]
        det = SimpleNamespace(color="red", shape="cube", top=np.array([0.3, 0.1, 0.8]))
        plan = policy._plan_pick(det, {})
        self.assertTrue(plan.ok, getattr(plan, "reason", ""))
        self.assertNotEqual(plan.skill_arm, "yaw_positive")
        self.assertEqual(plan.skill_arm, "yaw_negative")
        memory.close()

    def test_all_variants_infeasible_falls_back_to_stock_plan(self):
        memory = SkillMemory(str(self.root / "c.sqlite"), seed=0, log=lambda _: None)
        learn(memory, KEY, "yaw_positive", "stock")
        planner = Planner(infeasible={"yaw_positive", "yaw_negative"})
        policy = make_policy(memory, planner, retry=True)
        policy.vary_hist = [{"color": "red", "xy": np.array([0.3, 0.1]), "yaw": np.radians(45.0),  # stock (0 deg) must differ from the tried grasp
                             "posture": "closed_tight", "side": "left", "held": False}]
        det = SimpleNamespace(color="red", shape="cube", top=np.array([0.3, 0.1, 0.8]))
        plan = policy._plan_pick(det, {})
        self.assertTrue(plan.ok)
        self.assertEqual(plan.skill_arm, "stock")
        memory.close()

    def test_order_only_ever_contains_closed_set_members(self):
        memory = SkillMemory(str(self.root / "d.sqlite"), seed=5, log=lambda _: None)
        for arm in ARMS + ("invented_motion",):
            memory.record(KEY, arm, "placed", 0.0)
        for _ in range(20):
            order = memory.choose(KEY, ["stock", "invented_motion", "yaw_negative"])
            self.assertTrue(set(order) <= {"stock", "yaw_negative"}, order)
        memory.close()


class TestSqliteReload(unittest.TestCase):
    """(b) what is recorded to a SQLite file is loaded by a fresh instance and reorders identically."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_rows_written_then_reloaded_by_new_instance(self):
        path = str(self.root / "reload.sqlite")
        first = SkillMemory(path, seed=3, log=lambda _: None)
        learn(first, KEY, "alternate_posture", "stock", n=4)
        first.flush()
        first.close()
        with contextlib.closing(sqlite3.connect(path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM observations").fetchone()[0], 8)
        second = SkillMemory(path, seed=3, log=lambda _: None)
        self.assertEqual(second.summary()["cube|red|left"]["best_arm"], "alternate_posture")
        arms = ["stock", "alternate_posture"]
        self.assertEqual(second.choose(KEY, arms)[0], "alternate_posture")
        second.close()

    def test_policy_config_path_persists_between_policies(self):
        path = str(self.root / "policy.sqlite")
        cfg = load_config(overlay="none")
        cfg["skills"]["memory"] = True
        cfg["skills"]["memory_path"] = path
        logs = []
        with no_env_path():
            first = ClassicalPolicy(config=cfg, log=logs.append)
            learn(first.skill_memory, KEY, "yaw_positive", "stock", n=2)
            first.skill_memory.flush()
            first.skill_memory.close()
            second = ClassicalPolicy(config=cfg, log=logs.append)
            self.assertEqual(second.skill_memory.summary()["cube|red|left"]["best_arm"], "yaw_positive")
            second.skill_memory.close()
        self.assertTrue(Path(path).exists())
        self.assertFalse([line for line in logs if "[skills]" in line])


class TestFallbackNeverRaises(unittest.TestCase):
    """(c) unwritable path, locked DB, or a bad env override: the full API keeps working in memory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.logs = []

    def tearDown(self):
        for path in self.root.rglob("*"):
            if path.is_dir():
                path.chmod(0o700)
        self._tmp.cleanup()

    def exercise(self, memory):
        try:
            self.assertEqual(memory.choose(KEY, DEFAULT), DEFAULT)
            learn(memory, KEY, "yaw_positive", "stock", n=2)
            firsts = [memory.choose(KEY, DEFAULT)[0] for _ in range(50)]
            self.assertGreater(firsts.count("yaw_positive"), 25, firsts)
            memory.flush()
            self.assertEqual(memory.summary()["cube|red|left"]["counts"],
                             {"stock": 2, "yaw_positive": 2})
        finally:
            memory.close()
        self.assertEqual(len(self.logs), 1, self.logs)

    def test_env_override_to_unwritable_path_falls_back(self):
        bad = str(self.root / "no_dir" / "x.sqlite")
        with mock.patch.dict(os.environ, {"AALTO_SKILLS_DB": bad}):
            memory = SkillMemory(None, seed=0, log=self.logs.append)
            self.exercise(memory)
        self.assertFalse((self.root / "no_dir").exists())

    def test_locked_db_with_record_and_choose_does_not_raise(self):
        path = self.root / "locked.sqlite"
        with contextlib.closing(sqlite3.connect(str(path), isolation_level=None)) as holder:
            holder.execute("BEGIN EXCLUSIVE")
            with no_env_path():
                memory = SkillMemory(str(path), seed=0, log=self.logs.append)
                self.exercise(memory)

    def test_policy_with_bad_memory_path_still_acts(self):
        cfg = load_config(overlay="none")
        cfg["skills"]["memory"] = True
        cfg["skills"]["memory_path"] = str(self.root / "missing" / "s.sqlite")
        with no_env_path():
            policy = ClassicalPolicy(config=cfg, log=self.logs.append)
        self.assertIsNotNone(policy.skill_memory)
        self.assertFalse(policy.skill_memory._db_ok)
        policy.skill_memory.close()


def episode_digest(policy_builder, seed):
    """Action bytes of one full mock episode: sha256 over every action field, in order, plus the step count."""
    from .mock_world import make_mock_env
    env = make_mock_env(seed=seed)
    pol = policy_builder(env)
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"]})
    digest, steps = hashlib.sha256(), 0
    while True:
        action = pol.act(obs)
        for key in sorted(action):
            value = action[key]
            digest.update(key.encode())
            if isinstance(value, (bool, int, float, str, type(None))):
                digest.update(repr(value).encode())
            else:
                digest.update(np.ascontiguousarray(np.asarray(value, dtype=float)).tobytes())
        steps += 1
        obs, info = env.step(action)
        if info["episode_over"] or steps > 2000:
            break
    return digest.hexdigest(), steps


class TestFlagOffIdentity(unittest.TestCase):
    """(d) skills.memory OFF: no memory object, no file touched, identical actions to a run with no memory code."""

    def test_flag_off_matches_no_memory_path_and_ignores_existing_db(self):
        with tempfile.TemporaryDirectory() as tmp, no_env_path():
            root = Path(tmp)
            db = root / "populated.sqlite"
            seeded = SkillMemory(str(db), seed=0, log=lambda _: None)
            learn(seeded, ("cube", "red", None), "yaw_positive", "stock", n=3)
            seeded.flush()
            seeded.close()
            before = db.read_bytes()

            def off_default(env):
                return ClassicalPolicy(log=None, safety=env.filter.cfg)

            def off_with_db(env):
                cfg = load_config(overrides={"skills.memory": False, "skills.memory_path": str(db)})
                return ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg)

            def no_memory_code(env):
                # Strongest "memory absent": constructing SkillMemory anywhere would fail the episode.
                with mock.patch("aaltonomous.baselines.classical.skill_memory.SkillMemory",
                                side_effect=AssertionError("SkillMemory built with flag OFF")):
                    pol = ClassicalPolicy(log=None, safety=env.filter.cfg)
                self.assertIsNone(pol.skill_memory)
                return pol

            base = episode_digest(off_default, seed=0)
            again = episode_digest(off_default, seed=0)
            with_db = episode_digest(off_with_db, seed=0)
            absent = episode_digest(no_memory_code, seed=0)
            self.assertEqual(base, again)
            self.assertEqual(base, with_db)
            self.assertEqual(base, absent)
            self.assertEqual(db.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
