#!/usr/bin/env python3
"""Generate the cached narration library with ElevenLabs or an offline voice."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path

import yaml

DEFAULT_MODEL = "eleven_multilingual_v2"
ALLOWED_MODELS = ("eleven_multilingual_v2", "eleven_flash_v2_5", "eleven_flash_v2")


def load_entries(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    clips = data.get("clips") or {}
    if not isinstance(clips, dict):
        raise ValueError("clips.yaml must contain a 'clips' mapping")
    out = {}
    for key, value in clips.items():
        value = {"text": value} if isinstance(value, str) else dict(value or {})
        text = value.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"clip {key!r} needs non-empty text")
        out[str(key)] = value
    return out


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _http_post(url, headers, body, timeout=30.0):
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def elevenlabs_pcm(text, voice_id, model, api_key, http_post=_http_post):
    url = ("https://api.elevenlabs.io/v1/text-to-speech/"
           f"{voice_id}?output_format=pcm_16000")
    body = json.dumps({"text": text, "model_id": model}).encode("utf-8")
    headers = {"xi-api-key": api_key, "content-type": "application/json", "accept": "audio/pcm"}
    return http_post(url, headers, body)


def write_pcm_wav(path, pcm):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)


def offline_say(text, path, runner=subprocess.run):
    """Create a placeholder WAV with macOS say or espeak; never uses a network."""
    if shutil.which("say"):
        cmd = ["say", "--file-format=WAVE", "--data-format=LEI16@16000", "-o", str(path), text]
    else:
        exe = shutil.which("espeak") or shutil.which("espeak-ng")
        if not exe:
            raise RuntimeError("--say needs macOS say, espeak, or espeak-ng")
        cmd = [exe, "-s", "175", "-w", str(path), text]
    runner(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generate_all(config_path, output_dir, voice_id=None, model=DEFAULT_MODEL,
                 use_say=False, dry_run=False, api_key=None, http_post=_http_post,
                 say_fn=offline_say, log=print):
    """Generate changed clips and return ``(generated, skipped, planned)``.

    ``http_post`` and ``say_fn`` are injectable so tests remain entirely offline.
    """
    config_path, output_dir = Path(config_path), Path(output_dir)
    entries = load_entries(config_path)
    manifest_path = output_dir / "manifest.json"
    try:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        old = {"clips": {}}
    provider = "say" if use_say else "elevenlabs"
    if not use_say and not dry_run:
        api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is required (never put it in source or clips.yaml)")
        if not voice_id:
            raise RuntimeError("--voice-id is required for ElevenLabs generation")
    generated, skipped, planned, result = [], [], [], {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, entry in entries.items():
        if entry.get("generate", True) is False:
            continue
        text = entry["text"].strip()
        digest = text_hash(text)
        filename = key.replace(".", "_") + ".wav"
        meta = {"text": text, "text_sha256": digest, "provider": provider,
                "voice_id": voice_id if not use_say else None,
                "model_id": model if not use_say else None,
                "format": "pcm_s16le_16000_mono", "file": filename}
        previous = (old.get("clips") or {}).get(key) or {}
        comparable = ("text_sha256", "provider", "voice_id", "model_id", "format", "file")
        if all(previous.get(name) == meta.get(name) for name in comparable) and (output_dir / filename).is_file():
            result[key] = previous
            skipped.append(key)
            continue
        planned.append(key)
        if dry_run:
            log(f"would generate {key}: {text}")
            continue
        target = output_dir / filename
        temp = target.with_suffix(".tmp.wav")
        try:
            if use_say:
                say_fn(text, temp)
            else:
                pcm = elevenlabs_pcm(text, voice_id, model, api_key, http_post=http_post)
                write_pcm_wav(temp, pcm)
            temp.replace(target)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
        meta["generated_at"] = datetime.now(timezone.utc).isoformat()
        result[key] = meta
        generated.append(key)
        log(f"generated {key} -> {target}")
    if not dry_run:
        payload = {"version": 1, "source": str(config_path), "clips": result}
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir,
                                         prefix="manifest.", suffix=".tmp", delete=False) as tmp:
            json.dump(payload, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
            tmp_path = Path(tmp.name)
        tmp_path.replace(manifest_path)
    return generated, skipped, planned


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=here / "clips.yaml")
    ap.add_argument("--output-dir", type=Path, default=here / "clips")
    ap.add_argument("--voice-id", help="ElevenLabs voice ID (required unless --say or --dry-run)")
    ap.add_argument("--model", choices=ALLOWED_MODELS, default=DEFAULT_MODEL)
    ap.add_argument("--say", action="store_true", help="offline placeholders via macOS say or espeak")
    ap.add_argument("--dry-run", action="store_true", help="list stale/missing clips without generating")
    args = ap.parse_args(argv)
    try:
        generated, skipped, planned = generate_all(
            args.config, args.output_dir, args.voice_id, args.model, args.say, args.dry_run)
    except (RuntimeError, ValueError, OSError, urllib.error.URLError) as exc:
        ap.error(str(exc))
    print(f"generated={len(generated)} skipped={len(skipped)} planned={len(planned)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
