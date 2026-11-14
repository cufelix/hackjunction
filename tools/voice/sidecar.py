#!/usr/bin/env python3
"""Receive policy voice events and serialize playback of cached clips."""
from __future__ import annotations

import argparse
import json
import queue
import shutil
import socket
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path

import yaml

ALLOWED_EVENTS = {
    "episode_start", "instruction_understood", "picked", "placed",
    "abstained", "unknown", "episode_done",
}


def parse_event(data, max_bytes=1024):
    """Validate one UTF-8 JSON datagram and return a small event dict."""
    if not isinstance(data, (bytes, bytearray)) or len(data) > max_bytes:
        raise ValueError("invalid datagram size")
    obj = json.loads(bytes(data).decode("utf-8"))
    if not isinstance(obj, dict) or obj.get("event") not in ALLOWED_EVENTS:
        raise ValueError("unknown voice event")
    for key in ("colour", "shape", "bin", "reason", "summary"):
        if key in obj and (not isinstance(obj[key], str) or len(obj[key]) > 500):
            raise ValueError(f"invalid {key}")
    if "count" in obj and (not isinstance(obj["count"], int) or isinstance(obj["count"], bool)
                           or not 0 <= obj["count"] <= 100):
        raise ValueError("invalid count")
    return obj


def clip_keys(event):
    """Most-specific to generic cached clip keys for an event."""
    kind = event["event"]
    colour = str(event.get("colour", "unknown")).lower()
    shape = str(event.get("shape", "unknown")).lower()
    bin_name = str(event.get("bin", "unknown")).lower()
    if kind == "picked":
        return [f"picked.{colour}.{shape}", f"picked.{colour}", "picked"]
    if kind == "placed":
        return [f"placed.{colour}.{shape}.{bin_name}", f"placed.{bin_name}", "placed"]
    if kind == "episode_done":
        return [f"episode_done.{event.get('count', 0)}", "episode_done.default"]
    return [kind]


class ClipCatalog:
    def __init__(self, config_path, clips_dir=None, manifest_path=None):
        self.config_path = Path(config_path)
        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.entries = data.get("clips") or {}
        self.clips_dir = Path(clips_dir) if clips_dir else self.config_path.with_name("clips")
        manifest_path = Path(manifest_path) if manifest_path else self.clips_dir / "manifest.json"
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8")).get("clips", {})
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self.manifest = {}

    def resolve(self, event):
        for key in clip_keys(event):
            meta = self.manifest.get(key) or {}
            filename = meta.get("file")
            if filename:
                path = self.clips_dir / filename
                if path.is_file():
                    return key, path
            safe = key.replace(".", "_")
            for suffix in (".wav", ".mp3"):
                path = self.clips_dir / f"{safe}{suffix}"
                if path.is_file():
                    return key, path
        return None, None


class NullPlayer:
    """Log-only backend used for dry runs, tests, and silent finals."""

    def __init__(self, log=print):
        self.log = log or (lambda *_: None)
        self.played = []

    def play(self, path):
        self.played.append(Path(path))
        self.log(f"[voice] would play {path}")


class LocalPlayer:
    """PC playback through the first available aplay, ffplay, or afplay."""

    def __init__(self, command="auto"):
        commands = [command] if command != "auto" else ["aplay", "ffplay", "afplay"]
        self.command = next((c for c in commands if shutil.which(c)), None)
        if self.command is None:
            raise RuntimeError("no local player found (want aplay, ffplay, or afplay)")

    def play(self, path):
        cmd = ([self.command, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
               if self.command == "ffplay" else [self.command, str(path)])
        subprocess.run(cmd, check=False, stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class G1Player:
    """Unitree SDK2 AudioClient backend for 16 kHz mono PCM WAV clips."""

    def __init__(self, interface=None, volume=60):
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
        except ImportError as exc:
            raise RuntimeError("unitree_sdk2py with the G1 AudioClient is not installed") from exc
        ChannelFactoryInitialize(0, interface) if interface else ChannelFactoryInitialize(0)
        self.client = AudioClient()
        self.client.SetTimeout(10.0)
        self.client.Init()
        self.client.SetVolume(int(volume))

    def play(self, path):
        path = Path(path)
        if path.suffix.lower() != ".wav":
            raise ValueError("G1 backend requires WAV (16 kHz, mono, signed 16-bit PCM)")
        with wave.open(str(path), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                raise ValueError("G1 WAV must be 16 kHz mono signed 16-bit PCM")
            pcm = wav.readframes(wav.getnframes())
        stream_id = uuid.uuid4().hex
        for start in range(0, len(pcm), 32000):
            chunk = pcm[start:start + 32000]
            # SDK2 Python: AudioClient.PlayStream(app_name, stream_id, pcm_bytes).
            self.client.PlayStream("aaltonomous_voice", stream_id, chunk)
            time.sleep(len(chunk) / 32000.0)


class EventProcessor:
    """Bounded priority queue, duplicate suppression, and single-file playback."""

    PRIORITY = {"episode_done": 0, "unknown": 1, "abstained": 1}

    def __init__(self, catalog, player, queue_size=16, dedupe_s=3.0,
                 min_interval_s=0.35, clock=time.monotonic, sleeper=time.sleep, log=print):
        self.catalog, self.player = catalog, player
        self.queue = queue.PriorityQueue(maxsize=max(1, int(queue_size)))
        self.dedupe_s, self.min_interval_s = float(dedupe_s), float(min_interval_s)
        self.clock, self.sleeper = clock, sleeper
        self.log = log or (lambda *_: None)
        self._seen, self._seq, self._last_play = {}, 0, -float("inf")

    @staticmethod
    def signature(event):
        return tuple((key, json.dumps(value, sort_keys=True)) for key, value in sorted(event.items()) if key != "ts")

    def submit(self, event):
        now, sig = self.clock(), self.signature(event)
        if now - self._seen.get(sig, -float("inf")) < self.dedupe_s:
            return False
        self._seen[sig] = now
        self._seen = {key: at for key, at in self._seen.items() if now - at < max(30.0, self.dedupe_s)}
        self._seq += 1
        item = (self.PRIORITY.get(event["event"], 5), self._seq, event)
        try:
            self.queue.put_nowait(item)
            return True
        except queue.Full:
            if event["event"] != "episode_done":
                return False
            items = []
            try:
                while True:
                    items.append(self.queue.get_nowait())
                    self.queue.task_done()
            except queue.Empty:
                pass
            try:
                # Keep the best priorities; bounded queues are tiny (default 16).
                for kept in sorted(items + [item])[:self.queue.maxsize]:
                    self.queue.put_nowait(kept)
                return True
            except queue.Full:
                return False

    def play_one(self):
        try:
            _priority, _seq, event = self.queue.get_nowait()
        except queue.Empty:
            return False
        try:
            wait = self.min_interval_s - (self.clock() - self._last_play)
            if wait > 0:
                self.sleeper(wait)
            key, path = self.catalog.resolve(event)
            if path is None:
                self.log(f"[voice] no cached clip for {event}")
            else:
                self.player.play(path)  # one worker: clips never overlap
                self._last_play = self.clock()
                self.log(f"[voice] played {key}")
        except Exception as exc:
            self.log(f"[voice] playback failed: {type(exc).__name__}: {exc}")
        finally:
            self.queue.task_done()
        return True


def _worker(processor, stop):
    while not stop.is_set():
        if not processor.play_one():
            stop.wait(0.05)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=45821)
    ap.add_argument("--clips-yaml", type=Path, default=here / "clips.yaml")
    ap.add_argument("--clips-dir", type=Path, default=here / "clips")
    ap.add_argument("--backend", choices=("local", "g1", "null"), default="local")
    ap.add_argument("--player", default="auto", help="local backend: auto/aplay/ffplay/afplay")
    ap.add_argument("--interface", help="G1 DDS network interface, for example eth0")
    ap.add_argument("--volume", type=int, default=60)
    ap.add_argument("--queue-size", type=int, default=16)
    ap.add_argument("--dedupe-seconds", type=float, default=3.0)
    ap.add_argument("--min-interval", type=float, default=0.35)
    args = ap.parse_args(argv)

    catalog = ClipCatalog(args.clips_yaml, args.clips_dir)
    player = (NullPlayer() if args.backend == "null" else
              G1Player(args.interface, args.volume) if args.backend == "g1" else LocalPlayer(args.player))
    processor = EventProcessor(catalog, player, args.queue_size, args.dedupe_seconds, args.min_interval)
    stop = threading.Event()
    thread = threading.Thread(target=_worker, args=(processor, stop), daemon=True, name="voice-playback")
    thread.start()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(0.5)
    print(f"[voice] listening on udp://{args.host}:{args.port} ({args.backend})")
    try:
        while True:
            try:
                data, _peer = sock.recvfrom(2048)
                processor.submit(parse_event(data))
            except socket.timeout:
                pass
            except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
                print(f"[voice] ignored datagram: {exc}")
    except KeyboardInterrupt:
        return 0
    finally:
        stop.set()
        thread.join(timeout=1.0)
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
