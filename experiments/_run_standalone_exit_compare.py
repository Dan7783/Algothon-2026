#!/usr/bin/env python
"""Standalone (conflict-free) desired PnL: friend z-band vs OU optimal vs OU mean.

For each OU-improved pair (+ stable-8 overlap), simulate PairAgent alone under:
  - friend's z-band params
  - OU params + MERGED_OU_EXIT=optimal
  - OU params + MERGED_OU_EXIT=mean

Windows: train desired PnL on 31-750, OOS desired PnL on 750-1000 (verify only).
Does NOT optimize on OOS. Writes auto_tuning_results/_standalone_exit_compare.json.
"""
from __future__ import annotations

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "auto_tuning_results"
OUT = RESULTS / "_standalone_exit_compare.json"
COMM = 0.0001
LEG = 10_000.0

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
    "MERGED_ML_MODEL",
]


def clear_merged() -> None:
    for k in MERGED_KEYS:
        os.environ.pop(k, None)


def load_mod(ou_exit: str):
    clear_merged()
    os.environ["MERGED_OU_EXIT"] = ou_exit
    # Force reimport so OU_EXIT_MODE / PairAgent catch_up pick up env.
    for name in list(sys.modules):
        if name == "merged_friend_ou" or name.startswith("merged_friend_ou."):
            del sys.modules[name]
    return importlib.import_module("merged_friend_ou")


def load_prices():
    import latest_tuner

    prices, _nt, _ninst = latest_tuner.loadPrices("prices.txt")
    return np.asarray(prices, dtype=float)


def friend_zband_lookup():
    clear_merged()
    for name in list(sys.modules):
        if name == "friend_book_latest" or name.startswith("friend_book_latest."):
            del sys.modules[name]
    import friend_book_latest as fb

    out = {}
    for row in fb._ACTIVE_PAIR_PARAMS:
        y, x = row[0], row[1]
        out[(y, x)] = ["zband", y, x, *row[2:]]
    return out


def ou_rows_from_catalog():
    cat = json.loads((RESULTS / "merged_friend_ou_catalog.json").read_text(encoding="utf-8"))
    return [r for r in cat if r[0] == "ou"]


def key_pairs():
    meta = json.loads((RESULTS / "ou_stable_and_improved.json").read_text(encoding="utf-8"))
    improved = {(p["y"], p["x"], p["pair"]) for p in meta["improved_31"]}
    stable = {(p["y"], p["x"], p["pair"]) for p in meta["stable_8"]}
    return improved, stable


def build_cfg(mod, row):
    engine, y_symbol, x_symbol = row[0], row[1], row[2]
    idx = 0
    if engine == "ou":
        (
            q_beta,
            kalman_var_ewma_alpha,
            cusum_k,
            h,
            max_cov_inflate,
            cov_decay,
            ou_window,
            kappa_ewma_alpha,
            sigma_ewma_alpha,
        ) = row[3:]
        return mod.PairCfg(
            idx=idx,
            y=mod.I[y_symbol],
            x=mod.I[x_symbol],
            filt=mod.FiltCfg(
                q_beta=q_beta,
                ewma_lambda=0.97,
                kalman_var_ewma_alpha=kalman_var_ewma_alpha,
                cusum_k=cusum_k,
                h=h,
                max_cov_inflate=max_cov_inflate,
                cov_decay=cov_decay,
            ),
            trade=None,
            ou=mod.OUCfg(
                ou_window=int(ou_window),
                kappa_ewma_alpha=kappa_ewma_alpha,
                sigma_ewma_alpha=sigma_ewma_alpha,
            ),
            engine="ou",
        )
    (
        q_beta,
        cusum_k,
        h,
        max_cov_inflate,
        cov_decay,
        z_entry,
        z_exit,
        z_stop,
    ) = row[3:]
    return mod.PairCfg(
        idx=idx,
        y=mod.I[y_symbol],
        x=mod.I[x_symbol],
        filt=mod.FiltCfg(
            q_beta=q_beta,
            cusum_k=cusum_k,
            h=h,
            max_cov_inflate=max_cov_inflate,
            cov_decay=cov_decay,
        ),
        trade=mod.TradeCfg(z_entry=z_entry, z_exit=z_exit, z_stop=z_stop),
        ou=mod.OUCfg(),
        engine="zband",
    )


def simulate_desired(mod, row, prices, start: int, end: int) -> float:
    """Desired (conflict-free) PnL over days [start, end) using positions at t earning t->t+1."""
    agent = mod.PairAgent(build_cfg(mod, row))
    y_i, x_i = agent.spec.y, agent.spec.x
    shares = np.zeros(2, dtype=float)
    pnl = 0.0
    # Warm state from day 0; accumulate PnL only in [start, end).
    for t in range(0, end - 1):
        agent.catch_up(prices[:, : t + 1])
        pos_y, pos_x, _ = agent.dollar_positions()
        px_y = max(float(prices[y_i, t]), 1e-12)
        px_x = max(float(prices[x_i, t]), 1e-12)
        new = np.array([pos_y / px_y, pos_x / px_x], dtype=float)
        if t >= start:
            dpx = np.array(
                [
                    float(prices[y_i, t + 1] - prices[y_i, t]),
                    float(prices[x_i, t + 1] - prices[x_i, t]),
                ]
            )
            turn = abs(new[0] - shares[0]) * px_y + abs(new[1] - shares[1]) * px_x
            pnl += float(np.dot(new, dpx) - COMM * turn)
        shares = new
    return pnl


def main() -> None:
    prices = load_prices()
    friend_zb = friend_zband_lookup()
    ou_rows = {(r[1], r[2]): r for r in ou_rows_from_catalog()}
    improved, stable = key_pairs()
    targets = {}
    for y, x, name in improved | stable:
        targets[name] = (y, x, "improved" if (y, x, name) in improved else "stable")
        if (y, x, name) in improved and (y, x, name) in stable:
            targets[name] = (y, x, "both")

    # Prefer catalog OU row orientation; fall back to improved_31 params via catalog only.
    print(f"targets={len(targets)} ou_catalog={len(ou_rows)}", flush=True)

    # zband path ignores OU_EXIT; optimal module is fine for zband sims.
    mod_opt = load_mod("optimal")
    mod_mean = load_mod("mean")
    # reload optimal after mean so helper names stay consistent if reused
    mod_opt = load_mod("optimal")
    rows_out = []
    wins = {"ou_opt_train": 0, "ou_mean_train": 0, "zb_train": 0}
    sum_zb_tr = sum_opt_tr = sum_mean_tr = 0.0
    sum_zb_oos = sum_opt_oos = sum_mean_oos = 0.0
    n_compared = 0

    for name, (y, x, cohort) in sorted(targets.items()):
        ou_row = ou_rows.get((y, x))
        zb_row = friend_zb.get((y, x)) or friend_zb.get((x, y))
        if zb_row is None:
            print(f"  SKIP {name}: no friend zband", flush=True)
            continue
        if ou_row is None:
            print(f"  SKIP {name}: no OU catalog row (cohort={cohort})", flush=True)
            continue

        zb_tr = simulate_desired(mod_opt, zb_row, prices, 31, 750)
        zb_oos = simulate_desired(mod_opt, zb_row, prices, 750, 1000)
        opt_tr = simulate_desired(mod_opt, ou_row, prices, 31, 750)
        opt_oos = simulate_desired(mod_opt, ou_row, prices, 750, 1000)
        # mean module must be current for OU mean-exit catch_up
        clear_merged()
        os.environ["MERGED_OU_EXIT"] = "mean"
        # PairAgent reads module-level OU_EXIT_MODE; use mod_mean which was
        # imported under mean. Re-bind in case GC/order issues.
        if getattr(mod_mean, "OU_EXIT_MODE", None) != "mean":
            mod_mean = load_mod("mean")
        mean_tr = simulate_desired(mod_mean, ou_row, prices, 31, 750)
        mean_oos = simulate_desired(mod_mean, ou_row, prices, 750, 1000)

        best_engine = max(
            [("zband", zb_tr), ("ou_optimal", opt_tr), ("ou_mean", mean_tr)],
            key=lambda t: t[1],
        )[0]
        if best_engine == "ou_optimal":
            wins["ou_opt_train"] += 1
        elif best_engine == "ou_mean":
            wins["ou_mean_train"] += 1
        else:
            wins["zb_train"] += 1

        sum_zb_tr += zb_tr
        sum_opt_tr += opt_tr
        sum_mean_tr += mean_tr
        sum_zb_oos += zb_oos
        sum_opt_oos += opt_oos
        sum_mean_oos += mean_oos
        n_compared += 1

        rec = {
            "pair": name,
            "cohort": cohort,
            "train_31_750": {
                "zband": zb_tr,
                "ou_optimal": opt_tr,
                "ou_mean": mean_tr,
                "best": best_engine,
                "delta_opt_vs_zb": opt_tr - zb_tr,
                "delta_mean_vs_zb": mean_tr - zb_tr,
            },
            "oos_750_1000_verify": {
                "zband": zb_oos,
                "ou_optimal": opt_oos,
                "ou_mean": mean_oos,
                "delta_opt_vs_zb": opt_oos - zb_oos,
                "delta_mean_vs_zb": mean_oos - zb_oos,
            },
        }
        rows_out.append(rec)
        print(
            f"  {name:12} train zb={zb_tr:.0f} opt={opt_tr:.0f} mean={mean_tr:.0f} "
            f"best={best_engine} | oos zb={zb_oos:.0f} opt={opt_oos:.0f} mean={mean_oos:.0f}",
            flush=True,
        )

    report = {
        "criterion": "standalone desired PnL; train 31-750 primary; OOS verify only",
        "n_compared": n_compared,
        "wins_train": wins,
        "sums_train_31_750": {
            "zband": sum_zb_tr,
            "ou_optimal": sum_opt_tr,
            "ou_mean": sum_mean_tr,
            "delta_opt_vs_zb": sum_opt_tr - sum_zb_tr,
            "delta_mean_vs_zb": sum_mean_tr - sum_zb_tr,
        },
        "sums_oos_750_1000_verify": {
            "zband": sum_zb_oos,
            "ou_optimal": sum_opt_oos,
            "ou_mean": sum_mean_oos,
            "delta_opt_vs_zb": sum_opt_oos - sum_zb_oos,
            "delta_mean_vs_zb": sum_mean_oos - sum_zb_oos,
        },
        "engine_ahead_standalone_train": (sum_opt_tr > sum_zb_tr) or (sum_mean_tr > sum_zb_tr),
        "pairs": rows_out,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)
    print(
        f"ENGINE_AHEAD_TRAIN={report['engine_ahead_standalone_train']} "
        f"sum_train zb={sum_zb_tr:.0f} opt={sum_opt_tr:.0f} mean={sum_mean_tr:.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
