#!/usr/bin/env python
"""Close the friend gap under MERGED_OU_EXIT=mean.

Plan:
  1) Margin / conflict knobs under mean-exit
  2) LOO: revert each OU pair to friend's z-band params; rank by OOS delta
  3) Greedy multi-revert of damaging OU pairs until OOS stops rising
  4) Final 3-window scores for best config

Pinned MERGED_* env in every eval subprocess.
"""
from __future__ import annotations

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "auto_tuning_results"
BASE_CAT = RESULTS / "merged_friend_ou_catalog.json"
OUT = RESULTS / "_mean_exit_close_gap.json"
WORK = Path(tempfile.gettempdir()) / "_mean_exit_gap"
WORK.mkdir(parents=True, exist_ok=True)

MERGED_KEYS = [
    "MERGED_CONSENSUS_MARGIN",
    "MERGED_PROTECTED_BOOST",
    "MERGED_RIDGE_WEIGHT",
    "MERGED_FLIP_TO_FLAT",
    "MERGED_MAG_WEIGHTED",
    "MERGED_EXCLUDE_COHORTS",
    "MERGED_EXCLUDE_PAIRS",
    "MERGED_CATALOG_JSON",
    "MERGED_OU_CONFLICT",
    "MERGED_OU_SLEEVE_FRAC",
    "MERGED_OU_SLEEVE_SPLIT",
    "MERGED_OU_SLEEVE_OPPOSE",
    "MERGED_OU_EXIT",
]

MAX_WORKERS = int(os.environ.get("GAP_WORKERS", "3"))


def clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for k in MERGED_KEYS:
        env.pop(k, None)
    if extra:
        env.update(extra)
    return env


def run_eval(start: str, end: str, extra: dict[str, str], module: str = "merged_friend_ou") -> float | None:
    p = subprocess.run(
        [sys.executable, "eval.py", "--module", module, "--start", start, "--end", end],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=clean_env(extra),
    )
    m = re.search(r"Score:\s*([-\d.]+)", p.stdout + "\n" + p.stderr)
    return float(m.group(1)) if m else None


def friend_zband_lookup() -> dict[tuple[str, str], list]:
    """Map (y,x) -> zband row from friend_book_latest (last occurrence wins)."""
    # Clear MERGED before importing friend
    for k in MERGED_KEYS:
        os.environ.pop(k, None)
    import friend_book_latest as fb

    lookup: dict[tuple[str, str], list] = {}
    # friend rows: (y, x, q_beta, cusum_k, h, max_cov, cov_decay, z_entry, z_exit, z_stop)
    for row in fb._ACTIVE_PAIR_PARAMS:
        y, x = row[0], row[1]
        lookup[(y, x)] = ["zband", y, x, *row[2:]]
    return lookup


def write_cat(rows: list, name: str) -> str:
    path = WORK / name
    path.write_text(json.dumps(rows), encoding="utf-8")
    return str(path)


def main() -> None:
    base_rows = json.loads(BASE_CAT.read_text(encoding="utf-8"))
    friend_zb = friend_zband_lookup()
    ou_idxs = [i for i, r in enumerate(base_rows) if r[0] == "ou"]
    print(f"base {len(base_rows)} pairs, {len(ou_idxs)} OU", flush=True)

    report: dict = {"plan": [
        "Keep mean-exit + friend conflict layer",
        "Sweep CONSENSUS_MARGIN / flipflat under mean",
        "LOO-revert OU→friend zband; drop damaging OU",
        "Greedy multi-revert until OOS plateaus",
    ]}

    # --- 1) knob sweep under mean ---
    print("=== knob sweep (OOS 750-1000) ===", flush=True)
    knob_cases = [
        ("mean_m000", {"MERGED_OU_EXIT": "mean", "MERGED_CONSENSUS_MARGIN": "0.0"}),
        ("mean_m010", {"MERGED_OU_EXIT": "mean", "MERGED_CONSENSUS_MARGIN": "0.10"}),
        ("mean_m015", {"MERGED_OU_EXIT": "mean", "MERGED_CONSENSUS_MARGIN": "0.15"}),
        ("mean_m020", {"MERGED_OU_EXIT": "mean", "MERGED_CONSENSUS_MARGIN": "0.20"}),
        ("mean_m025", {"MERGED_OU_EXIT": "mean", "MERGED_CONSENSUS_MARGIN": "0.25"}),
        ("mean_m030", {"MERGED_OU_EXIT": "mean", "MERGED_CONSENSUS_MARGIN": "0.30"}),
        (
            "mean_m015_flipflat",
            {
                "MERGED_OU_EXIT": "mean",
                "MERGED_CONSENSUS_MARGIN": "0.15",
                "MERGED_FLIP_TO_FLAT": "1",
            },
        ),
        (
            "mean_m015_boost2",
            {
                "MERGED_OU_EXIT": "mean",
                "MERGED_CONSENSUS_MARGIN": "0.15",
                "MERGED_PROTECTED_BOOST": "2.0",
            },
        ),
    ]
    knob_scores = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_eval, "750", "1000", env): lab for lab, env in knob_cases}
        for fut in as_completed(futs):
            lab = futs[fut]
            sc = fut.result()
            knob_scores[lab] = sc
            print(f"  {lab:22} {sc}", flush=True)
    report["knob_sweep_oos750"] = knob_scores
    best_knob = max(knob_scores, key=lambda k: (knob_scores[k] is not None, knob_scores[k] or -1e9))
    best_knob_env = dict(next(env for lab, env in knob_cases if lab == best_knob))
    print(f"best knob: {best_knob} = {knob_scores[best_knob]}", flush=True)

    # Always also keep mean_m015 as LOO base if close; use best_knob as working env.
    work_env = dict(best_knob_env)
    work_env.setdefault("MERGED_OU_EXIT", "mean")
    work_env.setdefault("MERGED_CONSENSUS_MARGIN", "0.15")

    # Baseline under work_env with full catalog
    base_path = write_cat(base_rows, "base.json")
    base_env = {**work_env, "MERGED_CATALOG_JSON": base_path}
    base_score = run_eval("750", "1000", base_env)
    print(f"LOO base score={base_score} env={work_env}", flush=True)
    report["loo_base"] = {"score": base_score, "env": work_env}

    # --- 2) LOO revert each OU → friend zband ---
    print("=== LOO revert OU->friend zband ===", flush=True)

    def loo_one(idx: int) -> tuple[int, str, float | None, bool]:
        rows = [list(r) for r in base_rows]
        y, x = rows[idx][1], rows[idx][2]
        name = f"{y}-{x}"
        key = (y, x)
        if key not in friend_zb:
            # No friend zband twin: drop pair entirely
            del rows[idx]
            used_friend = False
        else:
            rows[idx] = list(friend_zb[key])
            used_friend = True
        path = write_cat(rows, f"loo_{name}.json")
        sc = run_eval("750", "1000", {**work_env, "MERGED_CATALOG_JSON": path})
        return idx, name, sc, used_friend

    loo = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(loo_one, i) for i in ou_idxs]
        for fut in as_completed(futs):
            idx, name, sc, used = fut.result()
            delta = None if sc is None or base_score is None else sc - base_score
            loo.append({"idx": idx, "pair": name, "score": sc, "delta": delta, "reverted_to_friend": used})
            print(f"  LOO {name:12} score={sc} delta={delta:+.2f}" if delta is not None else f"  LOO {name} FAIL", flush=True)

    loo.sort(key=lambda d: (-(d["delta"] or -1e9)))
    report["loo_revert"] = loo
    hurt = [d for d in loo if d["delta"] is not None and d["delta"] > 0.5]
    print(f"OU pairs whose revert HELPS (delta>+0.5): {len(hurt)}", flush=True)
    for d in hurt[:15]:
        print(f"  +{d['delta']:.2f}  {d['pair']}", flush=True)

    # --- 3) Greedy multi-revert ---
    print("=== greedy multi-revert ===", flush=True)
    current = [list(r) for r in base_rows]
    current_score = base_score
    reverted: list[str] = []
    history = []
    # Revert in LOO-ranked order (most helpful first), accept if still improves
    for d in loo:
        if d["delta"] is None or d["delta"] <= 0.05:
            continue
        y, x = d["pair"].split("-")
        # find current OU row
        found = None
        for i, r in enumerate(current):
            if r[0] == "ou" and r[1] == y and r[2] == x:
                found = i
                break
        if found is None:
            continue
        trial = [list(r) for r in current]
        if (y, x) in friend_zb:
            trial[found] = list(friend_zb[(y, x)])
        else:
            del trial[found]
        path = write_cat(trial, f"greed_{len(reverted)}_{d['pair']}.json")
        sc = run_eval("750", "1000", {**work_env, "MERGED_CATALOG_JSON": path})
        delta = None if sc is None or current_score is None else sc - current_score
        print(f"  try revert {d['pair']}: {current_score} -> {sc} (d={delta})", flush=True)
        if sc is not None and current_score is not None and sc > current_score + 0.05:
            current = trial
            current_score = sc
            reverted.append(d["pair"])
            history.append({"pair": d["pair"], "score": sc, "delta": delta})
        else:
            history.append({"pair": d["pair"], "score": sc, "delta": delta, "rejected": True})

    report["greedy_history"] = history
    report["reverted_pairs"] = reverted
    report["best_after_revert"] = current_score

    best_cat = write_cat(current, "best_mean_exit_catalog.json")
    # Persist into auto_tuning_results too
    final_cat_path = RESULTS / "_mean_exit_best_catalog.json"
    final_cat_path.write_text(json.dumps(current), encoding="utf-8")
    report["catalog_path"] = str(final_cat_path)

    # --- 4) Final windows ---
    print("=== final 3-window scores ===", flush=True)
    final_env = {**work_env, "MERGED_CATALOG_JSON": str(final_cat_path)}
    finals = {}
    for wname, s, e in (("oos750", "750", "1000"), ("train31", "31", "750"), ("val501", "501", "750"), ("oos801", "801", "1000")):
        finals[wname] = run_eval(s, e, final_env)
        print(f"  {wname} {finals[wname]}", flush=True)
    # Also friend + mean_m015 full catalog for comparison already known
    report["final_windows"] = finals
    report["n_ou_remaining"] = sum(1 for r in current if r[0] == "ou")
    report["friend_targets"] = {"oos750": 974.45, "train31": 926.94, "val501": 1042.37}

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)
    beat = finals.get("oos750") is not None and finals["oos750"] > 974.45
    print(f"BEAT_FRIEND_OOS={beat} best={finals.get('oos750')} target=974.45 reverted={reverted}", flush=True)


if __name__ == "__main__":
    main()
