#!/usr/bin/env python3
"""Thin wrapper so `python scripts/bench_match.py ...` == `python bench/bench_match.py ...`."""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "bench" / "bench_match.py"
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
