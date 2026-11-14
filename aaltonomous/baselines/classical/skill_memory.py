"""Persistent grasp-variant skill memory.

The closed arm set is exactly the retry-vary motion family from ``policy.py``:
``stock``, ``alternate_posture``, ``yaw_positive``, ``yaw_negative``,
``alternate_posture_yaw_positive``, and ``alternate_posture_yaw_negative``.
The yaw magnitude remains the existing shape-specific ``policy.retry_yaw_deg``
value; this module only orders those variants and never creates a motion.
"""
from __future__ import annotations

from collections import defaultdict
import contextlib
import logging
import os
import queue
import random
import sqlite3
import threading
import time


ARMS = (
    "stock",
    "alternate_posture",
    "yaw_positive",
    "yaw_negative",
    "alternate_posture_yaw_positive",
    "alternate_posture_yaw_negative",
)
OUTCOMES = ("held", "miss", "dropped", "placed")
REWARDS = {"held": 0.3, "miss": 0.0, "dropped": 0.0, "placed": 1.0}


class SkillMemory:
    """Beta-Bernoulli Thompson sampler with asynchronous SQLite persistence."""

    def __init__(self, path=None, seed=0, log=None):
        # In-memory unless a path is configured (AALTO_SKILLS_DB wins). No path = no file, no thread.
        self.path = os.environ.get("AALTO_SKILLS_DB") or path or None
        self.rng = random.Random(seed)
        self.log = log
        self._lock = threading.Lock()
        self._stats = defaultdict(lambda: defaultdict(self._empty_stats))
        self._warned = False
        self._db_ok = self.path is not None
        self._writes = queue.Queue()
        self._writer = None
        if self.path is not None:
            problem = self._path_problem()
            if problem is None:
                self._load()
            else:
                self._disable_db(OSError(problem))
        if self._db_ok:
            self._writer = threading.Thread(target=self._write_loop, name="skill-memory-db", daemon=True)
            self._writer.start()

    @staticmethod
    def _empty_stats():
        return {"n": 0, "reward": 0.0, "duration_s": 0.0,
                "outcomes": {name: 0 for name in OUTCOMES}}

    @staticmethod
    def _normalise_key(key):
        if not isinstance(key, (tuple, list)) or len(key) != 3:
            raise ValueError("skill key must be (shape, colour_family, hand)")
        return tuple("unknown" if value is None else str(value) for value in key)

    def reseed(self, seed):
        """Start the episode's deterministic Thompson-sampling stream."""
        self.rng.seed(0 if seed is None else int(seed))

    def _path_problem(self):
        """Why the configured path cannot hold a DB (missing/unwritable directory, or a directory), else None."""
        path = os.path.abspath(self.path)
        directory = os.path.dirname(path)
        if os.path.isdir(path):
            return f"{path} is a directory"
        if not os.path.isdir(directory):
            return f"directory {directory} does not exist"
        if not os.access(directory, os.W_OK | os.X_OK):
            return f"directory {directory} is not writable"
        return None

    def _connect(self):
        return sqlite3.connect(self.path, timeout=0.05)

    def _load(self):
        try:
            with contextlib.closing(self._connect()) as db:   # sqlite3's own `with` commits but never closes
                db.execute("PRAGMA busy_timeout=50")
                db.execute("""CREATE TABLE IF NOT EXISTS observations (
                    shape TEXT NOT NULL, colour_family TEXT NOT NULL, hand TEXT NOT NULL,
                    arm TEXT NOT NULL, outcome TEXT NOT NULL, reward REAL NOT NULL,
                    duration_s REAL NOT NULL, created_s REAL NOT NULL
                )""")
                rows = db.execute("""SELECT shape, colour_family, hand, arm, outcome,
                                             reward, duration_s FROM observations""").fetchall()
            for shape, colour, hand, arm, outcome, reward, duration in rows:
                if arm in ARMS and outcome in OUTCOMES:
                    self._add((shape, colour, hand), arm, outcome, float(reward), float(duration))
        except Exception as exc:
            self._disable_db(exc)

    def _disable_db(self, exc):
        self._db_ok = False
        if self._warned:
            return
        self._warned = True
        message = f"[skills] SQLite unavailable ({type(exc).__name__}: {exc}); using in-memory skill data"
        try:
            if self.log is not None:
                self.log(message)
            else:
                logging.getLogger(__name__).warning(message)
        except Exception:
            pass

    def _add(self, key, arm, outcome, reward, duration_s):
        stat = self._stats[key][arm]
        stat["n"] += 1
        stat["reward"] += reward
        stat["duration_s"] += duration_s
        stat["outcomes"][outcome] += 1

    def record(self, key, arm, outcome, duration_s=0.0):
        """Record one terminal attempt outcome; the DB write never runs on the caller's thread."""
        try:
            key = self._normalise_key(key)
            if arm not in ARMS:
                return
            if outcome not in OUTCOMES:
                raise ValueError(f"unknown skill outcome {outcome!r}")
            duration_s = max(0.0, float(duration_s))
            reward = REWARDS[outcome]
            with self._lock:
                self._add(key, arm, outcome, reward, duration_s)
            if self._db_ok:
                self._writes.put_nowait((key, arm, outcome, reward, duration_s, time.time()))
        except Exception as exc:
            # Bad persistence or logging must never escape into policy.act(). Invalid outcomes are
            # programmer errors too, but the control loop remains fail-open as required.
            self._disable_db(exc)

    def choose(self, key, default_order):
        """Return known arms in Thompson-sampled order, preserving defaults before two observations."""
        try:
            key = self._normalise_key(key)
            order = []
            for arm in default_order:
                if arm in ARMS and arm not in order:
                    order.append(arm)
            if not order:
                return []
            with self._lock:
                stats = {arm: dict(self._stats[key][arm]) for arm in order}
            if sum(stat["n"] for stat in stats.values()) < 2:
                return order
            sampled = []
            for index, arm in enumerate(order):
                stat = stats[arm]
                alpha = (4.0 if index == 0 else 1.0) + stat["reward"]
                beta = 1.0 + stat["n"] - stat["reward"]
                sampled.append((self.rng.betavariate(alpha, beta), -index, arm))
            sampled.sort(reverse=True)
            return [arm for _, _, arm in sampled]
        except Exception as exc:
            self._disable_db(exc)
            return [arm for arm in default_order if arm in ARMS]

    def summary(self):
        """Return the empirical best arm and compact arm/outcome counts for each observed key."""
        with self._lock:
            snapshot = {key: {arm: {"n": stat["n"], "reward": stat["reward"],
                                    "outcomes": dict(stat["outcomes"])}
                              for arm, stat in arms.items() if stat["n"]}
                        for key, arms in self._stats.items()}
        out = {}
        for key, arms in snapshot.items():
            if not arms:
                continue
            best = max(arms, key=lambda arm: (arms[arm]["reward"] / arms[arm]["n"],
                                              -ARMS.index(arm)))
            name = "|".join(key)
            out[name] = {"best_arm": best,
                         "counts": {arm: stat["n"] for arm, stat in arms.items()},
                         "outcomes": {arm: stat["outcomes"] for arm, stat in arms.items()}}
        return out

    def _write_loop(self):
        db = None
        try:
            db = self._connect()
            db.execute("PRAGMA busy_timeout=50")
            while True:
                item = self._writes.get()
                try:
                    if item is None:
                        return
                    key, arm, outcome, reward, duration_s, created_s = item
                    db.execute("INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                               (*key, arm, outcome, reward, duration_s, created_s))
                    db.commit()
                except Exception as exc:
                    self._disable_db(exc)
                finally:
                    self._writes.task_done()
        except Exception as exc:
            self._disable_db(exc)
            while True:
                try:
                    item = self._writes.get_nowait()
                except queue.Empty:
                    break
                self._writes.task_done()
                if item is None:
                    break
        finally:
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass

    def flush(self):
        """Wait for queued writes (tests/episode boundaries; never called from the 30 Hz path)."""
        self._writes.join()

    def close(self):
        if self._writer is not None and self._writer.is_alive():
            self._writes.put(None)
            self._writer.join(timeout=0.2)

