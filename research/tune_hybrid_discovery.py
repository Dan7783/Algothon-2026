"""Discovery-sleeve knob sweep for robust_hybrid (two-window Pareto rule).

The existing audit swept ridge entry, extra-LL removal and discovery scale.
This sweeps the remaining discovery knobs — entry z, break z, fit window —
including the later-entry variants suggested by the trade-timing analysis.
A variant replaces the default only if it improves BOTH train and test
score (the user-sanctioned two-window selection); otherwise defaults stay.
"""

from __future__ import annotations

# --- repo path bootstrap: run as `python research/<script>.py` from root ---
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "research"), str(_REPO_ROOT / "archive")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------

import json
from pathlib import Path

import pandas as pd

import robust_hybrid as strategy
from research.audit_robust_hybrid import configure, replay

OUT = Path("cusum_output/robust_hybrid/discovery_sweep.json")

VARIANTS = {
    "default e0.2 b4.5 w250": dict(entry=0.2, brk=4.5, window=250),
    "entry 0.4": dict(entry=0.4, brk=4.5, window=250),
    "entry 0.75": dict(entry=0.75, brk=4.5, window=250),
    "break 3.5": dict(entry=0.2, brk=3.5, window=250),
    "break 6.0": dict(entry=0.2, brk=6.0, window=250),
    "window 150": dict(entry=0.2, brk=4.5, window=150),
}


def main() -> None:
    prices = pd.read_csv("prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    report = {}
    for name, cfg in VARIANTS.items():
        configure(discovery=True, extra=True, ridge=1.0, scale=1.0)
        strategy.DISCOVERY_ENTRY_Z = cfg["entry"]
        strategy.DISCOVERY_BREAK_Z = cfg["brk"]
        strategy.DISCOVERY_FIT_WINDOW = cfg["window"]
        train = replay(prices, 251, 500)
        test = replay(prices, 501, 750)
        report[name] = {"config": cfg, "train": train, "test": test}
        print(
            f"{name:24s} train {train['score']:8.2f} ({train['mean']:7.2f})  "
            f"test {test['score']:8.2f} ({test['mean']:7.2f})"
        )
    strategy.DISCOVERY_ENTRY_Z = 0.2
    strategy.DISCOVERY_BREAK_Z = 4.5
    strategy.DISCOVERY_FIT_WINDOW = 250
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
