"""Measure why each fixed atlas pair is inactive under current teamName gates."""

# --- repo path bootstrap: run as `python research/<script>.py` from root ---
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "research"), str(_REPO_ROOT / "archive")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ---------------------------------------------------------------------------

from pathlib import Path
import types

import numpy as np
import pandas as pd


root = Path(__file__).resolve().parents[1]
prices_df = pd.read_csv(root / "prices.txt", sep=r"\s+")
names = prices_df.columns.to_list()
log_prices = np.log(prices_df.to_numpy(float).T)

source = root / "teamName.py"
strategy = types.ModuleType("teamName_source")
strategy.__file__ = str(source)
exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), strategy.__dict__)

rows = []
for i, j in strategy.ATLAS_PAIRS:
    daily = []
    for day in range(249, 749):
        history = log_prices[:, :day + 1]
        beta = strategy._kalman_beta(history[i, :-1], history[j, :-1])
        spread = history[i] - beta * history[j]
        adf_t = float(strategy._adf1_tstats(spread.reshape(-1, 1))[0])
        recent = spread[-145:]
        z = float((recent[-1] - recent[:-1].mean()) / (recent[:-1].std() + 1e-12))
        kappa = float(strategy._ou_kappa(spread[-120:]))
        gates = {
            "adf": adf_t < -2.9,
            "kappa": kappa > 0.0,
            "not_dead": abs(z) >= 0.27,
            "not_broken": abs(z) <= 4.0,
        }
        daily.append(gates)

    frame = pd.DataFrame(daily)
    active = frame.all(axis=1)
    row = {
        "pair": f"{names[i]}~{names[j]}",
        "active_days": int(active.sum()),
        "active_pct": float(100 * active.mean()),
    }
    for gate in frame.columns:
        row[f"{gate}_pass_pct"] = float(100 * frame[gate].mean())
        others = [name for name in frame.columns if name != gate]
        row[f"{gate}_sole_block_days"] = int((~frame[gate] & frame[others].all(axis=1)).sum())
    rows.append(row)

result = pd.DataFrame(rows).sort_values("active_pct")
out = root / "cointegration_output" / "atlas_pair_spreads" / "atlas_gate_failure_breakdown.csv"
result.to_csv(out, index=False)
print(result.to_string(index=False, float_format=lambda value: f"{value:6.1f}"))
print(f"\nSaved {out}")
