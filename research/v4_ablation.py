#!/usr/bin/env python
"""Attribute nontestbedv4's score to the two mechanisms ported from four_sigmas.

Runs the same engine and the same 88-pair catalog four ways:

  greedy      testbedv3's original exclusive-take allocation (the baseline the
              overlap vote replaces)
  pairs only  consensus overlap vote, ridge disabled
  ridge only  ridge sleeve, pair votes disabled
  vote        both (the shipped v4 default)

Reports every 125/250-day block so the both-train-blocks rule can be applied.

Run from the repo root:  python research/v4_ablation.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BLOCKS = [("trainA", 251, 375), ("trainB", 376, 500),
          ("val", 501, 750), ("org", 751, 1000)]

VARIANTS = {
    "greedy":     {"V4_ALLOCATION_MODE": "greedy"},
    "pairs only": {"V4_ALLOCATION_MODE": "vote", "V4_USE_RIDGE": "0"},
    "ridge only": {"V4_ALLOCATION_MODE": "vote", "V4_USE_PAIR": "0"},
    "vote (both)": {"V4_ALLOCATION_MODE": "vote"},
}


def run(module: str, lo: int, hi: int, env_extra: dict) -> float:
    env = dict(os.environ)
    env.pop("V4_USE_PAIR", None)
    env.pop("V4_USE_RIDGE", None)
    env.pop("V4_ALLOCATION_MODE", None)
    env.update(env_extra)
    out = subprocess.run(
        [sys.executable, "eval.py", "--module", module,
         "--start", str(lo), "--end", str(hi)],
        cwd=ROOT, env=env, capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("Score:"):
            return float(line.split(":")[1])
    return float("nan")


def main() -> None:
    print(f"{'variant':>13} " + "".join(f"{b[0]:>10}" for b in BLOCKS)
          + f"{'floor':>10}")
    rows = {}
    for name, env in VARIANTS.items():
        vals = [run("nontestbedv4", lo, hi, env) for _, lo, hi in BLOCKS]
        rows[name] = vals
        print(f"{name:>13} " + "".join(f"{v:10.1f}" for v in vals)
              + f"{min(vals):10.1f}")

    fs = [run("four_sigmas", lo, hi, {}) for _, lo, hi in BLOCKS]
    print(f"{'four_sigmas':>13} " + "".join(f"{v:10.1f}" for v in fs)
          + f"{min(fs):10.1f}")

    print("\ndelta of the ported overlap vote vs testbedv3's exclusive-take:")
    for (name, lo, hi), a, b in zip(BLOCKS, rows["vote (both)"], rows["greedy"]):
        print(f"  {name:>7}: {a - b:+8.1f}")


if __name__ == "__main__":
    main()
