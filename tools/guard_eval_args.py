#!/usr/bin/env python3
"""Reject evaluator arguments that would leave the default oracle policy active."""
from __future__ import annotations

import sys


def _has(args, name):
    return name in args or any(arg.startswith(name + "=") for arg in args)


def check(args):
    if _has(args, "--container") and not _has(args, "--policy"):
        return "refusing --container without an explicit --policy (the evaluator default is oracle)"
    return None


def main(argv=None):
    message = check(list(sys.argv[1:] if argv is None else argv))
    if message:
        print(f"guard_eval_args: {message}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
