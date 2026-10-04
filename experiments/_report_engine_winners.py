"""List catalog pairs where the tuned OU engine beats the z-band engine on desired OOS PnL."""
# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import json
import pathlib
import re

att = json.load(open("auto_tuning_results/_pair_attribution.json"))
tuned = {r["pair"]: r for r in att["tuned"]["records"]}
hybrid = {r["pair"]: r for r in att["hybrid"]["records"]}

rows = []
for name, tr in tuned.items():
    hz = hybrid.get(name)
    if hz and tr["cohort"] == "catalog" and hz["engine"] == "zband":
        rows.append((name, tr["pnl_oos"], hz["pnl_oos"], tr["pnl_full"], hz["pnl_full"]))

winners = [r for r in rows if r[1] > r[2]]
winners.sort(key=lambda r: r[1] - r[2], reverse=True)
print(f"catalog pairs improved by new tuning (desired OOS, OU vs zband): {len(winners)}/{len(rows)}\n")
print(f"{'pair':14} {'ou_oos':>8} {'zb_oos':>8} {'delta':>8} | {'ou_full':>8} {'zb_full':>8}")
for name, to, hz_o, tf, hz_f in winners:
    print(f"{name:14} {to:8.0f} {hz_o:8.0f} {to-hz_o:+8.0f} | {tf:8.0f} {hz_f:8.0f}")
print(f"\nsum delta winners: {sum(r[1]-r[2] for r in winners):+.0f}")
losers = [r for r in rows if r[1] <= r[2]]
print(f"sum delta losers:  {sum(r[1]-r[2] for r in losers):+.0f}  (n={len(losers)})")
