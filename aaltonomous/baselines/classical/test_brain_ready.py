"""Brain readiness on an open instruction: a down brain holds the arms home and re-probes instead of ending the
episode; a healthy brain resumes the look; act() stays non-blocking throughout."""
import time
import unittest

from aaltonomous.baselines.classical.brain import LlamaCppHTTPBackend
from aaltonomous.baselines.classical.test_brain_agent import ON, Agent, bin_of, build, pan_steps

DEAD = "http://127.0.0.1:9/completion"
OPEN = "Put the pan in the left bin."
DEAD_ON = {**ON, "brain.backend": "llamacpp", "brain.url": DEAD}


class Flaky(Agent):
    """Down (health False, every call raises) until .up is set; then answers like Agent."""

    def __init__(self, steps):
        super().__init__(steps)
        self.up = False

    def healthy(self, timeout=0.5):
        return self.up

    def decide(self, request_json):
        if not self.up:
            raise ConnectionError("brain down")
        return super().decide(request_json)

    def plan(self, request_json):
        if not self.up:
            raise ConnectionError("brain down")
        return super().plan(request_json)


def run(env, pol, obs, until_t, on_step=None, cap=4000):
    """act() loop like test_brain_agent.drive, stopping at sim time until_t (or the episode end)."""
    dts, n = [], 0
    while n < cap:
        t0 = time.perf_counter()
        act = pol.act(obs)
        dts.append(time.perf_counter() - t0)
        obs, info = env.step(act)
        n += 1
        if on_step is not None:
            on_step(float(obs["t"]), act)
        end = time.monotonic() + 5.0
        while (getattr(pol, "_job", None) or {}).get("result", True) is None and time.monotonic() < end:
            time.sleep(0.0005)
        if info["episode_over"] or float(obs["t"]) >= until_t:
            break
    return obs, info, dts


class TestBrainReady(unittest.TestCase):
    def test_unreachable_brain_keeps_the_episode_alive_and_arms_home(self):
        env, pol, obs, lines, _ = build(LlamaCppHTTPBackend(DEAD), OPEN, DEAD_ON)
        probes = []
        orig = pol.brain.probe
        pol.brain.probe = lambda *a, **k: (probes.append(1), orig(*a, **k))[1]
        home_err, waiting = [], []

        def watch(t, act):
            if pol.state == "brain_wait":
                waiting.append(t)
                arm = act["arm_q_target"][-1]
                home_err.append(float(abs(arm - pol.home).max()))

        obs, info, _ = run(env, pol, obs, until_t=10.0, on_step=watch)
        self.assertFalse(info["episode_over"], lines[-8:])
        self.assertTrue(waiting, "brain_wait never entered on an open instruction with the brain down")
        self.assertLess(max(home_err), 0.1, "arms left home while waiting for the brain")
        self.assertGreaterEqual(len(probes), 3, "entry probe plus re-probes every down_retry_s (3 s) while waiting")
        self.assertTrue(any("brain down" in l for l in lines), lines[-8:])

    def test_brain_that_becomes_healthy_mid_wait_is_resumed(self):
        be = Flaky(pan_steps())
        env, pol, obs, lines, pan = build(be, OPEN, DEAD_ON)
        pol.brain.backend = be                 # the BrainLayer's own backend: same mock as the planner adapter
        pol.brain.worker.backend = be
        flipped = []

        def flip(t, act):
            if t >= 5.0 and not flipped:
                be.up = True
                flipped.append(t)

        obs, info, dts = run(env, pol, obs, until_t=120.0, on_step=flip, cap=3500)
        self.assertTrue(flipped, "the wait never reached the healthy flip")
        self.assertTrue(any("brain healthy again" in l for l in lines), lines[-8:])
        self.assertTrue(any(p.ok for p in pol.plans), [(p.color, p.ok, p.reason) for p in pol.plans][-6:])
        self.assertEqual(bin_of(env, pan["pos"]), "left")
        p95 = sorted(dts)[int(0.95 * len(dts)) - 1]
        self.assertLess(p95, 0.03, f"act() p95 {p95 * 1000:.1f} ms")


if __name__ == "__main__":
    unittest.main()
