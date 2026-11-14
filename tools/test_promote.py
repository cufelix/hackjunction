"""Tests for tools/promote.sh.

Every test runs the script inside a throwaway git repo (a copy of this tree, committed), so the real
worktree is never touched. Run with the project venv:

    PYTHON=<venv python> python -m pytest tools/test_promote.py

Equivalence is skipped here (--no-equiv) to keep the tests fast; the full run is a manual step.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
OVL = "aaltonomous/baselines/classical/overlays"
SUB = f"{OVL}/submission.yaml"
GIT = ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run([*GIT, *args], cwd=repo, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A committed, clean copy of this tree (no .git from the worktree; a fresh repo instead)."""
    dst = tmp_path / "repo"
    shutil.copytree(REPO, dst, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    git(dst, "init", "-q")
    git(dst, "add", "-A")
    git(dst, "commit", "-q", "-m", "test: snapshot")
    return dst


def promote(repo: Path, *args: str, tmp: Path | None = None):
    env = dict(os.environ, PYTHON=sys.executable)
    if tmp is not None:
        env["TMPDIR"] = str(tmp)   # keep the script's temp dirs inside pytest's tmp_path
    return subprocess.run(["bash", str(repo / "tools/promote.sh"), *args], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=600)


def dry_output_dir(stdout: str) -> Path:
    m = re.search(r"^dry-run output dir: (\S+)$", stdout, re.M)
    assert m, f"no dry-run output dir in:\n{stdout}"
    return Path(m.group(1))


def test_dry_run_rc1_writes_release_md(repo, tmp_path):
    before = sha256(repo / SUB)
    res = promote(repo, "--dry-run", "--no-equiv", "rc1", tmp=tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr

    out = dry_output_dir(res.stdout)
    rel = (out / "RELEASE.md").read_text()
    copy_sub = out / "tree" / SUB

    # The promoted overlay: header line, name convention, content from rc1.
    head = copy_sub.read_text().splitlines()
    sha = git(repo, "rev-parse", "--short", "HEAD").strip()
    assert f"# promoted from rc1 @ {sha} on " in copy_sub.read_text()
    assert "name: submission" in head
    assert "instruction.normalize: true" in copy_sub.read_text()

    # RELEASE.md content.
    full = git(repo, "rev-parse", "HEAD").strip()
    assert f"- Git SHA: `{full}`" in rel
    assert "- Overlay promoted: `rc1`" in rel
    assert "Flags ON:" in rel
    assert "| `instruction.normalize` | `true` |" in rel
    assert "`perception.hsv_set`" in rel
    assert f"`{SUB}`: `{sha256(copy_sub)}`" in rel
    cfg_hash = sha256(out / "tree/aaltonomous/baselines/classical/config.yaml")
    assert f"`aaltonomous/baselines/classical/config.yaml`: `{cfg_hash}`" in rel
    assert "Verdict: **PASS**" in rel
    assert "RELEASE GATE PASS: submission == rc1" in rel
    assert "SKIPPED (`--no-equiv`)" in rel
    assert "docker build -f tools/submission.Dockerfile -t hackjunction/policy:<tag> ." in rel

    # Dry run changes nothing in the repo.
    assert sha256(repo / SUB) == before
    assert git(repo, "status", "--porcelain", "--untracked-files=all") == ""


def test_dry_run_finals_r21_checks_reviewed_allow_list(repo, tmp_path):
    res = promote(repo, "--dry-run", "--no-equiv", "finals_r21", tmp=tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "RELEASE FLAGS PASS" in res.stdout


def test_unknown_overlay_fails_cleanly(repo, tmp_path):
    before = sha256(repo / SUB)
    res = promote(repo, "--dry-run", "--no-equiv", "does_not_exist", tmp=tmp_path)
    assert res.returncode != 0
    assert "no overlay 'does_not_exist'" in res.stderr
    assert "Traceback" not in res.stdout + res.stderr
    assert sha256(repo / SUB) == before
    assert not list(tmp_path.glob("promote-dry.*"))


def test_dirty_tree_is_refused(repo, tmp_path):
    (repo / "README.md").write_text((repo / "README.md").read_text() + "\nlocal edit\n")
    res = promote(repo, "--dry-run", "--no-equiv", "rc1", tmp=tmp_path)
    assert res.returncode != 0
    assert "git tree is dirty" in res.stderr
    assert "README.md" in res.stderr


def test_real_run_writes_then_rerun_is_allowed(repo, tmp_path):
    res = promote(repo, "--no-equiv", "rc1", tmp=tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "NOT committed" in res.stdout
    assert "git add" in res.stdout and "git commit" in res.stdout
    assert "name: submission" in (repo / SUB).read_text()
    assert (repo / "RELEASE.md").exists()
    assert not (repo / "RELEASE.equivalence.txt").exists()
    status = git(repo, "status", "--porcelain", "--untracked-files=all").splitlines()
    assert status, "expected the release files to be modified"
    assert {line[3:] for line in status} <= {SUB, "RELEASE.md", "RELEASE.flags.json"}

    # The files this script writes may already be dirty: a second run must not be refused.
    res2 = promote(repo, "--no-equiv", "rc1", tmp=tmp_path)
    assert res2.returncode == 0, res2.stdout + res2.stderr


def test_failed_gate_restores_tree(repo, tmp_path):
    # An unknown flag makes the candidate unloadable: the gate fails after submission.yaml was written.
    bad = repo / OVL / "bad_candidate.yaml"
    bad.write_text("name: bad_candidate\nflags:\n  no.such.flag: true\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "test: bad candidate")
    before = (repo / SUB).read_bytes()

    res = promote(repo, "--no-equiv", "bad_candidate", tmp=tmp_path)
    assert res.returncode != 0
    assert "release gate FAILED" in res.stderr
    assert (repo / SUB).read_bytes() == before
    assert not (repo / "RELEASE.md").exists()
    assert git(repo, "status", "--porcelain", "--untracked-files=all") == ""
