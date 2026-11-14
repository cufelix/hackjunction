"""Best-effort voice events for the optional out-of-process audio sidecar.

The policy control path only calls :meth:`VoiceEventEmitter.emit`: one JSON
encoding and one non-blocking localhost UDP datagram.  There are deliberately
no retries, locks, file reads, subprocesses, SDK calls, or network services in
this module.  A missing listener and a full socket buffer both mean "drop".
"""
from __future__ import annotations

import json
import socket
import time


EVENTS = frozenset({
    "episode_start",
    "instruction_understood",
    "picked",
    "placed",
    "abstained",
    "unknown",
    "episode_done",
})


class VoiceEventEmitter:
    """Non-blocking UDP event sink; every failure is intentionally ignored."""

    def __init__(self, host="127.0.0.1", port=45821, send_buffer_bytes=8192,
                 max_datagram_bytes=1024, sock=None):
        self.address = (str(host), int(port))
        self.max_datagram_bytes = max(128, int(max_datagram_bytes))
        self.sock = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setblocking(False)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF,
                                 max(1024, int(send_buffer_bytes)))
        except (AttributeError, OSError, ValueError):
            pass

    def emit(self, event, **fields):
        """Send one event or drop it.  This method never raises or waits."""
        try:
            if event not in EVENTS:
                return False
            payload = {"event": event, "ts": round(time.time(), 3)}
            payload.update(fields)
            data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            if len(data) > self.max_datagram_bytes:
                return False
            self.sock.sendto(data, self.address)
            return True
        except (BlockingIOError, InterruptedError, OSError, TypeError, ValueError):
            return False

    def close(self):
        try:
            self.sock.close()
        except (AttributeError, OSError):
            pass

