#!/usr/bin/env python
"""Greedy portfolio assembly under the live 88-pair friend_ou ship book.

Base: merged_friend_ou catalog (friend_latest + 35 OU), shared conflict,
CONSENSUS_MARGIN=0.15. Catalog hard-cap is 88 pairs, so candidates are
*swapped in* for removable base rows (not appended).

Protected + supplied-reintroduced pairs are never displaced.
Candidates in DEVELOPMENT_REMOVED / PREVIOUS_OOS_REMOVED are skipped.

Phase 0: leave-one-out on removable slots → rank worst contributors.
Phase 1: each candidate vs top-W worst slots (parallel); keep best swap delta>0.
Phase 2: sequential forward selection of improving swaps (two passes).

Judge (friend convention): eval.py OOS **750–1000** via isolated MERGED_* env
(no session leaks). Accepted swaps also log train 31–750 and val 501–750.
Expected base ~906.86 (shared m0.15); friend baseline ~974.45 on same OOS.
Candidates prefer the 34 OOS-positive pairs from ou_search_individual_summary.

Outputs: auto_tuning_results/_greedy_final_catalog.json + _greedy_history.json.

Legacy note: earlier append-greedy targeted merged_strategy hybrid (~781 OOS,
margin 0.30) which allows >88 rows. That path is retained only as documentation;
this script is the ship-relevant assembly. Prior wrong-window run used 801–1000.
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
import time
from concurrent.futures import ThreadPoolExecutor

PARAM_KEYS = (
    "q_beta", "kalman_var_ewma_alpha", "cusum_k", "h", "max_cov_inflate",
    "cov_decay", "ou_window", "kappa_ewma_alpha", "sigma_ewma_alpha",
)
RESULTS = "auto_tuning_results"
BASE_CATALOG = os.path.join(RESULTS, "merged_friend_ou_catalog.json")
MODULE = "merged_friend_ou"
MARGIN = "0.15"
PROTECTED_BOOST = "4.0"
OU_CONFLICT = "shared"
WORST_SLOTS = 10  # phase-1 swap targets from LOO ranking
# Keep low: each merged_friend_ou eval is RAM-heavy; 4-wide LOO got killed (exit -1).
MAX_WORKERS = int(os.environ.get("GREEDY_WORKERS", "2"))
# Friend windows (primary judge = OOS 750-1000)
OOS_START, OOS_END = "750", "1000"
TRAIN_START, TRAIN_END = "31", "750"
VAL_START, VAL_END = "501", "750"
WORK = os.path.join(tempfile.gettempdir(), "_greedy_friend_ou")
os.makedirs(WORK, exist_ok=True)

# Mirror merged_friend_ou catalog guards (names are orientation-sensitive in
# the module checks; we match unordered keys for candidate filtering / freeze).
PROTECTED = {
    "AENO-NWIG", "CTGI-EELT", "EORC-NGTE", "HETT-ULXY",
    "HUXZ-ACAC", "RTTH-NAYO", "SMAH-ILVX",
}
REINTRODUCED = {"ANSO-MHRM", "NGTE-ILVX", "SMAH-NGTE"}
BANNED = {
    "ALGO-HRET", "SRNA-ELLT", "ULXY-EAFC",  # development-removed
    "CCNS-FARS", "DIHO-RTTH", "EORC-SPLZ", "NAYO-FWWG", "SPLZ-FWWG",  # prev OOS
}

MERGED_KEYS = (
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
)

SCORE_RE = re.compile(r"Score:\s*(-?[\d.]+)")


def ukey_syms(y, x):
    return frozenset((y, x))


def ukey_name(name):
    y, x = name.split("-", 1)
    return ukey_syms(y, x)


def row_name(row):
    return f"{row[1]}-{row[2]}"


def friend_ou_base_rows():
    return [list(r) for r in json.load(open(BASE_CATALOG))]


def _clean_env(extra):
    env = os.environ.copy()
    for k in MERGED_KEYS:
        env.pop(k, None)
    env.update(extra)
    return env


def eval_score(rows, tag, start=OOS_START, end=OOS_END, keep_log=False):
    if len(rows) > 88:
        raise RuntimeError(f"catalog len {len(rows)} exceeds 88-pair ship cap")
    cat = os.path.join(WORK, f"cat_{tag}.json")
    with open(cat, "w") as f:
        json.dump(rows, f)
    env = _clean_env({
        "MERGED_CATALOG_JSON": cat,
        "MERGED_CONSENSUS_MARGIN": MARGIN,
        "MERGED_PROTECTED_BOOST": PROTECTED_BOOST,
        "MERGED_OU_CONFLICT": OU_CONFLICT,
    })
    p = subprocess.run(
        [sys.executable, "eval.py", "--module", MODULE,
         "--start", str(start), "--end", str(end)],
        capture_output=True, text=True, env=env, cwd=os.getcwd(),
    )
    m = SCORE_RE.search(p.stdout)
    if m is None:
        log = os.path.join(WORK, f"fail_{tag}.log")
        with open(log, "w") as f:
            f.write(p.stdout + "\n--- STDERR ---\n" + p.stderr)
        raise RuntimeError(f"eval failed for {tag}: see {log}")
    if keep_log:
        with open(os.path.join(WORK, f"eval_{tag}.log"), "w") as f:
            f.write(p.stdout)
    return float(m.group(1))


def eval_windows(rows, tag):
    """Primary OOS judge plus train/val diagnostics (friend windows)."""
    oos = eval_score(rows, f"{tag}_oos", OOS_START, OOS_END)
    train = eval_score(rows, f"{tag}_train", TRAIN_START, TRAIN_END)
    val = eval_score(rows, f"{tag}_val", VAL_START, VAL_END)
    return {"oos": oos, "train": train, "val": val}


def oos_positive_keys():
    """Unordered keys for the 34 standalone-OOS-positive OU search pairs."""
    path = os.path.join(RESULTS, "ou_search_individual_summary.json")
    data = json.load(open(path))
    keys = set()
    for p in data.get("all_pairs", []):
        if float(p.get("oos_score", 0) or 0) <= 0:
            continue
        y, x = p.get("y"), p.get("x")
        if y and x:
            keys.add(ukey_syms(y, x))
    return keys


def candidate_rows(base_rows):
    existing = {ukey_syms(r[1], r[2]) for r in base_rows}
    banned = {ukey_name(n) for n in BANNED}
    prefer = oos_positive_keys()
    out, skipped_exist, skipped_banned, skipped_nonpos = [], [], [], []
    for e in json.load(open(os.path.join(RESULTS, "ou_search_tuning_results.json"))):
        fp = e.get("global", {}).get("final_params")
        oos = e.get("global", {}).get("oos_metrics", {}).get("score", float("-inf"))
        if not fp:
            continue
        name = f"{e['y_symbol']}-{e['x_symbol']}"
        key = ukey_syms(e["y_symbol"], e["x_symbol"])
        if key in existing:
            skipped_exist.append(name)
            continue
        if key in banned:
            skipped_banned.append(name)
            continue
        # Prefer the 34 OOS-positive standalones; skip non-positive for swap budget.
        if key not in prefer:
            skipped_nonpos.append(name)
            continue
        row = ["ou", e["y_symbol"], e["x_symbol"]] + [fp[k] for k in PARAM_KEYS]
        out.append({
            "name": name, "row": row, "standalone_oos": oos,
            "oos_positive": True,
        })
    out.sort(key=lambda c: -c["standalone_oos"])
    return out, skipped_exist, skipped_banned, skipped_nonpos


def removable_indices(base_rows):
    frozen = {ukey_name(n) for n in (PROTECTED | REINTRODUCED)}
    out = []
    for i, r in enumerate(base_rows):
        if ukey_syms(r[1], r[2]) in frozen:
            continue
        out.append(i)
    return out


def swap_rows(base_rows, idx, new_row):
    rows = [list(r) for r in base_rows]
    rows[idx] = list(new_row)
    return rows


def main():
    t0 = time.time()
    base = friend_ou_base_rows()
    assert len(base) <= 88
    print(f"base: {MODULE} catalog ({len(base)} rows) "
          f"margin={MARGIN} conflict={OU_CONFLICT} [SWAP mode] "
          f"judge OOS {OOS_START}-{OOS_END}", flush=True)
    base_score = eval_score(base, "base", keep_log=True)
    print(f"[base] OOS {OOS_START}-{OOS_END} score {base_score:.2f} "
          f"(expect ~906.86) ({time.time()-t0:.0f}s)", flush=True)

    cands, skip_exist, skip_banned, skip_nonpos = candidate_rows(base)
    rem_idx = removable_indices(base)
    print(f"candidates: {len(cands)} OOS-positive "
          f"(skip exist={len(skip_exist)} banned={len(skip_banned)} "
          f"nonpos={len(skip_nonpos)}); "
          f"removable slots={len(rem_idx)}; catalog_cap=88", flush=True)

    history = {
        "mode": "swap",
        "base_module": MODULE,
        "base_catalog": BASE_CATALOG,
        "margin": MARGIN,
        "ou_conflict": OU_CONFLICT,
        "judge_oos": [int(OOS_START), int(OOS_END)],
        "train_window": [int(TRAIN_START), int(TRAIN_END)],
        "val_window": [int(VAL_START), int(VAL_END)],
        "base_score": base_score,
        "skipped_already_in_base": skip_exist,
        "skipped_banned": skip_banned,
        "skipped_non_oos_positive": skip_nonpos,
        "n_candidates": len(cands),
        "worst_slots": WORST_SLOTS,
        "phase0_loo": [],
        "phase1": [],
        "phase2": [],
        "accepted_window_scores": [],
    }

    # ---- phase 0: leave-one-out contribution of removable slots ----
    def loo(i):
        rows = [list(r) for r in base]
        dropped = rows.pop(i)
        # temporary undersize catalog is OK for LOO ranking only — but
        # protected/reintroduced subset checks require those names present.
        # Dropping a non-frozen pair keeps guards intact; length < 88 is fine.
        s = eval_score(rows, f"loo_{row_name(dropped)}")
        return {
            "idx": i,
            "name": row_name(dropped),
            "loo_score": s,
            "contribution": base_score - s,  # >0 means pair helps base
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        loo_rows = list(ex.map(loo, rem_idx))
    loo_rows.sort(key=lambda r: r["contribution"])  # worst first
    history["phase0_loo"] = loo_rows
    targets = loo_rows[:WORST_SLOTS]
    print(f"[phase0] LOO done; worst {len(targets)} slots: "
          f"{[(t['name'], round(t['contribution'], 2)) for t in targets]} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- phase 1: each candidate vs worst slots (flattened for parallelism) ----
    jobs = [(c, t) for c in cands for t in targets]

    def probe_one(job):
        c, t = job
        s = eval_score(swap_rows(base, t["idx"], c["row"]),
                       f"p1_{c['name']}_for_{t['name']}")
        return {
            **c,
            "replaces": t["name"],
            "replaces_idx": t["idx"],
            "p1_score": s,
            "p1_delta": s - base_score,
        }

    print(f"[phase1] {len(jobs)} swap probes with {MAX_WORKERS} workers",
          flush=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        raw = list(ex.map(probe_one, jobs))
    best_by_name = {}
    for rec in raw:
        prev = best_by_name.get(rec["name"])
        if prev is None or rec["p1_delta"] > prev["p1_delta"]:
            best_by_name[rec["name"]] = rec
    p1 = list(best_by_name.values())
    history["phase1"] = [
        {k: c[k] for k in (
            "name", "standalone_oos", "replaces", "replaces_idx",
            "p1_score", "p1_delta")}
        for c in p1
    ]
    survivors = sorted(
        (c for c in p1 if c["p1_delta"] > 0.0),
        key=lambda c: -c["p1_delta"],
    )
    print(f"[phase1] {len(survivors)}/{len(cands)} improving swaps "
          f"({time.time()-t0:.0f}s)", flush=True)
    for c in survivors[:10]:
        print(f"  +{c['p1_delta']:.2f}  {c['name']} replaces {c['replaces']} "
              f"-> {c['p1_score']:.2f}", flush=True)

    # ---- phase 2: sequential forward swaps, two passes ----
    current = [list(r) for r in base]
    cur_score = base_score
    used_slots = set()
    accepted = []
    remaining = list(survivors)

    for pass_no in (1, 2):
        improved = False
        still = []
        for c in remaining:
            # Prefer originally proposed slot if still free; else retry worst
            # free slots from current LOO ranking order.
            trial_slots = []
            if c["replaces_idx"] not in used_slots:
                trial_slots.append(c["replaces_idx"])
            for t in targets:
                if t["idx"] not in used_slots and t["idx"] not in trial_slots:
                    trial_slots.append(t["idx"])
            if not trial_slots:
                still.append(c)
                continue
            best_keep = None
            for idx in trial_slots:
                s = eval_score(swap_rows(current, idx, c["row"]),
                               f"p2_{pass_no}_{c['name']}_i{idx}")
                if s > cur_score and (best_keep is None or s > best_keep[0]):
                    best_keep = (s, idx)
            kept = best_keep is not None
            history["phase2"].append({
                "pass": pass_no,
                "name": c["name"],
                "kept": kept,
                "score": None if not kept else best_keep[0],
                "replaces_idx": None if not kept else best_keep[1],
                "replaces": None if not kept else row_name(current[best_keep[1]]),
            })
            if kept:
                s, idx = best_keep
                replaced = row_name(current[idx])
                current = swap_rows(current, idx, c["row"])
                cur_score = s
                used_slots.add(idx)
                assert len(current) <= 88
                wins = eval_windows(current, f"accept_{c['name']}")
                rec = {
                    **c,
                    "final_replaces": replaced,
                    "final_idx": idx,
                    "oos": wins["oos"],
                    "train": wins["train"],
                    "val": wins["val"],
                }
                accepted.append(rec)
                history["accepted_window_scores"].append({
                    "name": c["name"],
                    "replaces": replaced,
                    "oos_750_1000": wins["oos"],
                    "train_31_750": wins["train"],
                    "val_501_750": wins["val"],
                })
                history["phase2"][-1].update({
                    "train": wins["train"],
                    "val": wins["val"],
                    "oos": wins["oos"],
                })
                improved = True
                print(
                    f"[phase2.{pass_no}] + {c['name']} for {replaced}: "
                    f"oos={wins['oos']:.2f} train={wins['train']:.2f} "
                    f"val={wins['val']:.2f} ({time.time()-t0:.0f}s)",
                    flush=True,
                )
            else:
                still.append(c)
        remaining = still
        if not improved:
            break

    assert len(current) <= 88
    with open(os.path.join(RESULTS, "_greedy_final_catalog.json"), "w") as f:
        json.dump(current, f, indent=1)
    with open(os.path.join(RESULTS, "_greedy_history.json"), "w") as f:
        json.dump(history, f, indent=1)
    print(f"[done] final {len(current)} rows | OOS {OOS_START}-{OOS_END} "
          f"{cur_score:.2f} (base {base_score:.2f}, "
          f"+{cur_score - base_score:.2f}) "
          f"accepted={[a['name'] + '<-' + a['final_replaces'] for a in accepted]}")
    print("GREEDY COMPLETE")


if __name__ == "__main__":
    main()
