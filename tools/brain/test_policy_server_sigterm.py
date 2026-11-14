"""SIGTERM must stop the policy server cleanly (exit 0 within 3 s), as PID 1 in the brain container.

Run: python -m unittest tools/brain/test_policy_server_sigterm.py   (from the repo root)
Lives here, not in aaltonomous/deploy/test_deploy.py, to keep the change inside tools/brain/.
"""
import signal
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SigtermTest(unittest.TestCase):
    def test_sigterm_exits_zero_within_3s(self):
        port = free_port()
        proc = subprocess.Popen(
            [sys.executable, "-m", "aaltonomous.deploy.policy_server", "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        try:
            deadline = time.monotonic() + 20
            while True:  # wait until the websocket socket accepts connections
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                    break
                except OSError:
                    if proc.poll() is not None or time.monotonic() > deadline:
                        self.fail(f"server did not start: {proc.stderr.read()[-2000:]}")
                    time.sleep(0.1)
            proc.send_signal(signal.SIGTERM)
            try:
                code = proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.fail("server still running 3 s after SIGTERM")
            self.assertEqual(code, 0, proc.stderr.read()[-2000:])
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stderr.close()


if __name__ == "__main__":
    unittest.main()
