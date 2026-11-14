"""Skill-memory persistence: in-memory by default, SQLite only when configured and usable, never raises."""
import contextlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from .config import load_config
from .skill_memory import SkillMemory

KEY = ("cube", "red", "left")


@contextlib.contextmanager
def no_env_path():
    with mock.patch.dict(os.environ):
        os.environ.pop("AALTO_SKILLS_DB", None)
        yield


class TestSkillMemoryPersist(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.logs = []

    def tearDown(self):
        for path in self.root.rglob("*"):   # restore write bits so the temp tree can be removed
            if path.is_dir():
                path.chmod(0o700)
        self._tmp.cleanup()

    def test_default_is_in_memory_and_creates_no_file(self):
        before = set(self.root.iterdir())
        cwd = os.getcwd()
        with no_env_path(), mock.patch("tempfile.tempdir", str(self.root)):
            os.chdir(self.root)
            try:
                memory = SkillMemory(seed=1, log=self.logs.append)
                memory.record(KEY, "stock", "placed", 0.1)
                memory.flush()
                self.assertEqual(memory.summary()["cube|red|left"]["counts"], {"stock": 1})
                memory.close()
            finally:
                os.chdir(cwd)
        self.assertIsNone(memory.path)
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertEqual(list(self.root.rglob("*")), [])
        self.assertEqual(self.logs, [])

    def test_valid_dir_persists_across_instances(self):
        path = self.root / "skills.sqlite"
        with no_env_path():
            first = SkillMemory(str(path), seed=0, log=self.logs.append)
            for _ in range(2):
                first.record(KEY, "yaw_positive", "placed", 0.2)
            first.flush()
            first.close()
            second = SkillMemory(str(path), seed=0, log=self.logs.append)
            self.assertEqual(second.summary()["cube|red|left"]["counts"], {"yaw_positive": 2})
            second.close()
        self.assertTrue(path.exists())
        self.assertEqual(self.logs, [])

    def test_env_var_overrides_configured_path(self):
        env_path = self.root / "env.sqlite"
        with mock.patch.dict(os.environ, {"AALTO_SKILLS_DB": str(env_path)}):
            memory = SkillMemory(str(self.root / "cfg.sqlite"), log=self.logs.append)
            memory.record(KEY, "stock", "placed", 0.0)
            memory.flush()
            memory.close()
        self.assertEqual(memory.path, str(env_path))
        self.assertTrue(env_path.exists())
        self.assertFalse((self.root / "cfg.sqlite").exists())

    def assert_in_memory_fallback(self, memory):
        """Fallback contract: records still counted, exactly one log line, nothing raised."""
        for _ in range(3):
            memory.record(KEY, "stock", "miss", 0.1)
            memory.record(KEY, "alternate_posture", "placed", 0.1)
        memory.flush()
        self.assertEqual(memory.summary()["cube|red|left"]["counts"], {"stock": 3, "alternate_posture": 3})
        self.assertEqual(len(self.logs), 1, self.logs)
        self.assertIn("in-memory", self.logs[0])
        memory.close()

    def test_missing_directory_falls_back_in_memory(self):
        path = self.root / "no_such_dir" / "skills.sqlite"
        with no_env_path():
            memory = SkillMemory(str(path), log=self.logs.append)
            self.assert_in_memory_fallback(memory)
        self.assertFalse(path.parent.exists())

    def test_read_only_directory_falls_back_in_memory(self):
        if os.geteuid() == 0:
            self.skipTest("root ignores directory write bits")
        ro = self.root / "ro"
        ro.mkdir()
        ro.chmod(0o500)
        with no_env_path():
            memory = SkillMemory(str(ro / "skills.sqlite"), log=self.logs.append)
            self.assert_in_memory_fallback(memory)
        self.assertEqual(list(ro.iterdir()), [])

    def test_corrupt_file_falls_back_in_memory(self):
        bad = self.root / "corrupt.sqlite"
        bad.write_bytes(b"\x00garbage-not-a-database\xff" * 200)
        with no_env_path():
            memory = SkillMemory(str(bad), log=self.logs.append)
            self.assert_in_memory_fallback(memory)

    def test_locked_database_falls_back_in_memory(self):
        path = self.root / "locked.sqlite"
        with contextlib.closing(sqlite3.connect(str(path), isolation_level=None)) as holder:
            holder.execute("BEGIN EXCLUSIVE")
            with no_env_path():
                memory = SkillMemory(str(path), log=self.logs.append)
                self.assert_in_memory_fallback(memory)

    def test_policy_construction_with_memory_on_and_no_volume_is_ok(self):
        from .policy import ClassicalPolicy
        logs = []
        for memory_path in (None, str(self.root / "no_volume" / "skills.sqlite")):
            cfg = load_config(overlay="none")
            cfg["skills"]["memory"] = True
            cfg["skills"]["memory_path"] = memory_path
            with no_env_path():
                policy = ClassicalPolicy(config=cfg, log=logs.append)
            self.assertIsNotNone(policy.skill_memory)
            policy.skill_memory.close()
        self.assertFalse((self.root / "no_volume").exists())


if __name__ == "__main__":
    unittest.main()
