#!/usr/bin/env python
"""Build the OU-search candidate file: top gate-passing screened pairs that are
NOT already in the friend catalog (zband universe) and not already tuned.
"""

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import json

import friend_book
from select_pairs import SANITY, key_of

TOPN = 80

screen = json.load(open("auto_tuning_results/screen_train.json"))
records = screen["records"]  # pre-sorted by screen_score desc

friend_keys = {key_of((r[0], r[1])) for r in friend_book._ACTIVE_PAIR_PARAMS}
tuned_keys = {
    key_of((e["y_symbol"], e["x_symbol"]))
    for e in json.load(open("auto_tuning_results/screen_tuning_results.json"))
}

eligible = [r for r in records if all(r["gate_eval"]["gates"][g] for g in SANITY)]
fresh = [
    r for r in eligible
    if key_of(r["pair"]) not in friend_keys and key_of(r["pair"]) not in tuned_keys
]

picked = fresh[:TOPN]
out = []
for r in picked:
    y, x = r["pair"] if r["orient"] == "A" else r["pair"][::-1]
    out.append({"y": y, "x": x})

with open("auto_tuning_results/ou_search_pairs.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"eligible gate-passing: {len(eligible)}/1225 | fresh (not in friend88/tuned10): {len(fresh)}")
print(f"picked top {len(out)} -> auto_tuning_results/ou_search_pairs.json")
for r in picked[:60]:
    g = r["gate_eval"]
    print(f"  {r['pair'][0]}-{r['pair'][1]:<7} score {g['screen_score']:7.2f} "
          f"medHL {g['median_halflife']:5.1f}")
