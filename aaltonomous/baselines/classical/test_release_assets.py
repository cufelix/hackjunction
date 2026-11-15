"""Every runtime asset the submission image needs is COPYed by tools/submission.Dockerfile."""
import glob
import os
import re
import shlex
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLASSICAL = ROOT / "aaltonomous/baselines/classical"
DOCKERFILE = ROOT / "tools/submission.Dockerfile"
# HERE / "a" / "b" or ASSET / "a" style joins, as used by the classical modules.
JOIN = re.compile(r'\b(?:HERE|ASSET\w*)((?:\s*/\s*"[^"]+")+)')


def copy_lines():
    text = DOCKERFILE.read_text().replace("\\\n", " ").replace("${OVERLAY}", "submission")   # ARG default
    for line in text.splitlines():
        if line.startswith("COPY "):
            tokens = shlex.split(line)[1:]
            yield tokens[:-1], tokens[-1].rstrip("/")


def image_files():
    """Paths the image will contain: COPY sources expanded against the repo, placed at their destinations."""
    out = set()
    for srcs, dest in copy_lines():
        for src in srcs:
            for hit in glob.glob(str(ROOT / src)) or [str(ROOT / src)]:
                if os.path.isdir(hit):
                    for dp, _, names in os.walk(hit):
                        out.update(f"{dest}/{os.path.relpath(os.path.join(dp, n), hit)}" for n in names)
                elif os.path.isfile(hit):
                    out.add(f"{dest}/{os.path.basename(hit)}" if len(srcs) > 1 or src.endswith("/") else dest)
    return out


def covered(path, image):
    return path in image or any(i.startswith(path + "/") for i in image)


def referenced_assets():
    refs = {}
    for py in CLASSICAL.rglob("*.py"):
        if py.name.startswith("test_"):
            continue
        for m in JOIN.finditer(py.read_text()):
            parts = re.findall(r'"([^"]+)"', m.group(1))
            refs["/".join(["aaltonomous/baselines/classical", *parts])] = str(py.relative_to(ROOT))
    return refs


class ReleaseAssetCoverage(unittest.TestCase):
    def test_assets_dir_fully_copied(self):
        assets = CLASSICAL / "assets"
        files = {str(p.relative_to(ROOT)) for p in assets.rglob("*") if p.is_file()}
        self.assertTrue(files, "classical/assets is empty")
        missing = {f for f in files if not covered(f, image_files())}
        self.assertFalse(missing, f"assets not COPYed: {sorted(missing)}")

    def test_referenced_assets_copied(self):
        refs = referenced_assets()
        self.assertTrue(refs, "no HERE/ASSET joins found; scan regex is broken")
        image = image_files()
        missing = {p: src for p, src in refs.items() if not covered(p, image)}
        self.assertFalse(missing, f"loaded at runtime but not COPYed: {missing}")


if __name__ == "__main__":
    unittest.main()
