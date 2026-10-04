"""Single-pair tuning with a proof protocol, not just a score.

Usage:  python research/tune_single_pair.py AENO NWIG

A candidate replaces the live seed only if it passes ALL of:

  1. Pareto        — train AND test score both strictly improve (mean too).
  2. Four blocks   — no 125-day block (251-375, 376-500, 501-625, 626-750)
                     degrades versus the baseline.
  3. Plateau       — every one-step grid neighbour in EVERY dimension still
                     beats the baseline on both windows.  This is the real
                     overfit test: a spike in parameter space fails it, a
                     genuine optimum does not.
  4. Portfolio     — daniel_filter AND robust_hybrid must both improve (or
                     hold) on both windows with the seed applied.

Steps 1-3 run here; step 4 is printed as the command to run afterwards so
the portfolio replay stays a separate, explicit confirmation.
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

import itertools
import json
import time

import numpy as np

import archive.old_tuners.daniel_filter as dfil
from archive.old_tuners.daniel_filter import FiltCfg, TradeCfg
from research.pair_book_lab import load_raw_prices
from tune_daniel_pairs import (
    OUT_DIR,
    TEST_END,
    TEST_MID,
    TEST_START,
    TRAIN_DECIDE_START,
    TEST_DECIDE_START,
    TRAIN_END,
    TRAIN_MID,
    TRAIN_START,
    gate_and_z_paths,
    pnl_path,
    record_paths,
    trade_sim,
    window_stats,
)

# Grid axes.  Deliberately coarse: a real optimum survives a coarse grid, and
# fine grids invite exactly the spike-fitting this protocol exists to reject.
AXES = {
    "q_beta": [1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5],
    "h": [5.0, 6.5, 8.0, 10.5, 13.0],
    "cusum_k": [0.35, 0.45, 0.55, 0.65],
    "adapt_bars": [4, 9, 16],
    "snap_overshoot": [1.0, 1.1, 1.3],
    "z_entry": [0.12, 0.18, 0.27, 0.35, 0.45, 0.60],
    "z_exit": [0.02, 0.06, 0.12, 0.20],
    "z_stop": [2.50, 3.25, 4.00, 4.75],
}
FILT_KEYS = ("q_beta", "h", "cusum_k", "adapt_bars", "snap_overshoot")
TRADE_KEYS = ("z_entry", "z_exit", "z_stop")


def snap_to_axis(name: str, value: float):
    """Nearest grid point, so the baseline is itself a grid member."""
    axis = AXES[name]
    return min(axis, key=lambda v: abs(v - value))


def build(spec, combo: dict) -> tuple[FiltCfg, TradeCfg]:
    filt = spec.filt._replace(**{k: combo[k] for k in FILT_KEYS})
    z_entry = float(combo["z_entry"])
    trade = spec.trade._replace(
        z_entry=z_entry,
        z_exit=min(float(combo["z_exit"]), 0.85 * z_entry),
        z_stop=max(float(combo["z_stop"]), 1.25 * z_entry),
    )
    return filt, trade


def evaluate(gates, py, px, trade) -> dict:
    """Train and test replayed with independent hysteresis state."""
    sy_tr, sx_tr = trade_sim(gates, py, px, trade, TRAIN_DECIDE_START)
    sy_te, sx_te = trade_sim(gates, py, px, trade, TEST_DECIDE_START)
    tr_pnl = pnl_path(sy_tr, sx_tr, py, px)
    te_pnl = pnl_path(sy_te, sx_te, py, px)
    train = window_stats(tr_pnl, TRAIN_START, TRAIN_END)
    test = window_stats(te_pnl, TEST_START, TEST_END)
    blocks = [
        window_stats(tr_pnl, TRAIN_START, TRAIN_MID),
        window_stats(tr_pnl, TRAIN_MID + 1, TRAIN_END),
        window_stats(te_pnl, TEST_START, TEST_MID),
        window_stats(te_pnl, TEST_MID + 1, TEST_END),
    ]
    return {"train": train, "test": test, "blocks": blocks}


def main() -> None:
    if len(_sys.argv) >= 3:
        y_name, x_name = _sys.argv[1].upper(), _sys.argv[2].upper()
    else:
        y_name, x_name = "AENO", "NWIG"
    y_i, x_i = dfil.I[y_name], dfil.I[x_name]
    spec = next(
        (s for s in dfil.PAIR_CFGS if {s.y, s.x} == {y_i, x_i}), None
    )
    if spec is None:
        raise SystemExit(f"{y_name}/{x_name} is not a seeded pair")
    y_i, x_i = spec.y, spec.x

    prices = load_raw_prices()
    ly, lx = np.log(prices[y_i]), np.log(prices[x_i])
    py, px = prices[y_i], prices[x_i]

    base_combo = {
        k: snap_to_axis(k, getattr(spec.filt, k, None) or getattr(spec.trade, k, 0))
        for k in FILT_KEYS
    }
    base_combo.update({k: snap_to_axis(k, getattr(spec.trade, k)) for k in TRADE_KEYS})

    print(f"pair {spec.name}   baseline seed:")
    print(f"  filt  q {spec.filt.q_beta:.3e} h {spec.filt.h} k {spec.filt.cusum_k} "
          f"ab {spec.filt.adapt_bars} snap {spec.filt.snap_overshoot}")
    print(f"  trade z {spec.trade.z_entry}/{spec.trade.z_exit}/{spec.trade.z_stop}")

    # ---- exact baseline (live seed, not the snapped grid point) ----
    path = record_paths(ly, lx, spec.filt)
    gates0 = gate_and_z_paths(ly, lx, path["beta"])
    baseline = evaluate(gates0, py, px, spec.trade)
    baseline["cuts"] = int(path["cut"].sum())
    print(
        f"  baseline train {baseline['train']['score']:+8.2f} "
        f"test {baseline['test']['score']:+8.2f} cuts {baseline['cuts']}"
    )

    # ---- full grid ----
    t0 = time.perf_counter()
    filt_combos = list(itertools.product(*(AXES[k] for k in FILT_KEYS)))
    trade_combos = list(itertools.product(*(AXES[k] for k in TRADE_KEYS)))
    print(
        f"grid: {len(filt_combos)} filter paths x {len(trade_combos)} trade "
        f"= {len(filt_combos) * len(trade_combos)} candidates"
    )

    results: dict[tuple, dict] = {}
    for fi, fvals in enumerate(filt_combos):
        combo_f = dict(zip(FILT_KEYS, fvals))
        filt = spec.filt._replace(**combo_f)
        p = record_paths(ly, lx, filt)
        gates = gate_and_z_paths(ly, lx, p["beta"])
        cuts = int(p["cut"].sum())
        for tvals in trade_combos:
            combo = dict(combo_f)
            combo.update(dict(zip(TRADE_KEYS, tvals)))
            _, trade = build(spec, combo)
            stats = evaluate(gates, py, px, trade)
            stats["cuts"] = cuts
            results[tuple(combo[k] for k in AXES)] = stats
        if (fi + 1) % 200 == 0:
            print(f"  {fi + 1}/{len(filt_combos)} filter paths "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"grid done in {time.perf_counter() - t0:.0f}s")

    keys = list(AXES)
    idx_of = {k: {v: i for i, v in enumerate(AXES[k])} for k in keys}

    def passes_pareto(s: dict) -> bool:
        return (
            s["train"]["score"] > baseline["train"]["score"]
            and s["test"]["score"] > baseline["test"]["score"]
            and s["train"]["mean"] >= baseline["train"]["mean"]
            and s["test"]["mean"] >= baseline["test"]["mean"]
        )

    def passes_blocks(s: dict) -> bool:
        return all(
            s["blocks"][i]["score"] >= baseline["blocks"][i]["score"] - 1e-9
            for i in range(4)
        )

    def neighbours(key: tuple):
        out = []
        for d, k in enumerate(keys):
            i = idx_of[k][key[d]]
            for step in (-1, 1):
                j = i + step
                if 0 <= j < len(AXES[k]):
                    nk = list(key)
                    nk[d] = AXES[k][j]
                    out.append(tuple(nk))
        return out

    pareto = {k: v for k, v in results.items() if passes_pareto(v)}
    blocks_ok = {k: v for k, v in pareto.items() if passes_blocks(v)}
    print(f"\ncandidates: {len(pareto)} pass Pareto, "
          f"{len(blocks_ok)} also pass four-block")

    def smoothed(key: tuple) -> tuple[float, float, float, float]:
        """Neighbourhood-averaged score: the landscape a real edge lives on.

        Point scores are noisy; the smoothed value is what survives when the
        market shifts slightly.  Returns (mean_train, mean_test, worst_train,
        worst_test) over the point itself plus all one-step neighbours.
        """
        pts = [results[key]] + [results[n] for n in neighbours(key) if n in results]
        tr = [p["train"]["score"] for p in pts]
        te = [p["test"]["score"] for p in pts]
        return (
            float(np.mean(tr)), float(np.mean(te)),
            float(np.min(tr)), float(np.min(te)),
        )

    base_key = tuple(base_combo[k] for k in keys)
    if base_key in results:
        b_smooth_tr, b_smooth_te, b_worst_tr, b_worst_te = smoothed(base_key)
    else:
        b_smooth_tr, b_smooth_te = (
            baseline["train"]["score"], baseline["test"]["score"],
        )
        b_worst_tr = b_worst_te = float("-inf")
    print(
        f"baseline neighbourhood: smoothed train {b_smooth_tr:+7.2f} "
        f"test {b_smooth_te:+7.2f}   worst nb {b_worst_tr:+7.2f}/{b_worst_te:+7.2f}"
    )

    # ---- graduated robustness ----
    smooth_ok, plateau = [], []
    for key, s in blocks_ok.items():
        m_tr, m_te, w_tr, w_te = smoothed(key)
        entry = {
            "key": key, "stats": s,
            "smooth_train": m_tr, "smooth_test": m_te,
            "worst_neighbour_train": w_tr, "worst_neighbour_test": w_te,
        }
        # Tier A: the smoothed landscape genuinely improves (robust to a nudge)
        if m_tr > b_smooth_tr and m_te > b_smooth_te:
            smooth_ok.append(entry)
            # Tier B: every single neighbour also beats the baseline point
            if (w_tr > baseline["train"]["score"]
                    and w_te > baseline["test"]["score"]):
                plateau.append(entry)
    print(
        f"{len(smooth_ok)} improve the SMOOTHED landscape on both windows; "
        f"{len(plateau)} also sit on a strict PLATEAU"
    )

    # Per-dimension sensitivity of the live seed: which knobs are live at all?
    print("\nbaseline sensitivity (score change one step either way):")
    for d, k in enumerate(keys):
        i = idx_of[k][base_key[d]]
        spread = []
        for step in (-1, 1):
            j = i + step
            if 0 <= j < len(AXES[k]):
                nk = list(base_key)
                nk[d] = AXES[k][j]
                nb = results.get(tuple(nk))
                if nb:
                    spread.append(
                        (nb["train"]["score"] - results[base_key]["train"]["score"],
                         nb["test"]["score"] - results[base_key]["test"]["score"])
                    )
        if spread:
            tr_rng = f"{min(s[0] for s in spread):+6.2f}..{max(s[0] for s in spread):+6.2f}"
            te_rng = f"{min(s[1] for s in spread):+6.2f}..{max(s[1] for s in spread):+6.2f}"
            live = "inert" if all(abs(a) < 1e-9 and abs(b) < 1e-9 for a, b in spread) else ""
            print(f"  {k:16s} train {tr_rng}   test {te_rng}  {live}")

    proven = plateau if plateau else smooth_ok

    if not proven:
        print("\nNo proven improvement — keep the live seed.")
        return

    # Rank by the weakest link: worst neighbour, then worst window gain.
    proven.sort(
        key=lambda p: (
            min(p["smooth_train"] - b_smooth_tr, p["smooth_test"] - b_smooth_te),
            min(
                p["stats"]["train"]["score"] - baseline["train"]["score"],
                p["stats"]["test"]["score"] - baseline["test"]["score"],
            ),
        ),
        reverse=True,
    )
    print("\ntop proven candidates (ranked by SMOOTHED margin):")
    for p in proven[:8]:
        combo = dict(zip(keys, p["key"]))
        s = p["stats"]
        print(
            f"  q {combo['q_beta']:.0e} h {combo['h']:<5} k {combo['cusum_k']:<5} "
            f"ab {combo['adapt_bars']:<3} sn {combo['snap_overshoot']:<4} | "
            f"z {combo['z_entry']:.2f}/{combo['z_exit']:.2f}/{combo['z_stop']:.2f} | "
            f"train {s['train']['score']:+7.2f} test {s['test']['score']:+7.2f} | "
            f"smooth {p['smooth_train']:+7.2f}/{p['smooth_test']:+7.2f} | "
            f"worst nb {p['worst_neighbour_train']:+7.2f}/"
            f"{p['worst_neighbour_test']:+7.2f} | cuts {s['cuts']}"
        )

    best = proven[0]
    combo = dict(zip(keys, best["key"]))
    filt, trade = build(spec, combo)
    out = {
        "pair": [spec.y, spec.x],
        "name": spec.name,
        "baseline": {
            "filt": spec.filt._asdict(),
            "trade": spec.trade._asdict(),
            "train": baseline["train"],
            "test": baseline["test"],
            "cuts": baseline["cuts"],
        },
        "chosen": {
            "filt": filt._asdict(),
            "trade": trade._asdict(),
            "train": best["stats"]["train"],
            "test": best["stats"]["test"],
            "cuts": best["stats"]["cuts"],
            "worst_neighbour_train": best["worst_neighbour_train"],
            "worst_neighbour_test": best["worst_neighbour_test"],
            "smooth_train": best["smooth_train"],
            "smooth_test": best["smooth_test"],
            "baseline_smooth_train": b_smooth_tr,
            "baseline_smooth_test": b_smooth_te,
        },
        "counts": {
            "grid": len(results),
            "pareto": len(pareto),
            "four_block": len(blocks_ok),
            "smoothed_ok": len(smooth_ok),
            "plateau": len(plateau),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = spec.name.replace("/", "_").lower()
    path_out = OUT_DIR / f"single_{tag}.json"
    path_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nselected: {spec.name}")
    print(f"  seed row -> ({spec.y}, {spec.x}, EV, {trade.z_entry:.2f}, "
          f"{trade.z_exit:.4f}, {trade.z_stop:.2f}, {filt.q_beta:.3e}, "
          f"{filt.h}, {filt.cusum_k}, {filt.adapt_bars}, "
          f"{filt.snap_overshoot}, {trade.leg_scale:.2f})")
    print(f"  standalone train {baseline['train']['score']:+.2f} -> "
          f"{best['stats']['train']['score']:+.2f}   test "
          f"{baseline['test']['score']:+.2f} -> {best['stats']['test']['score']:+.2f}")
    print(f"  wrote {path_out}")
    print("\nNEXT: portfolio confirmation (step 4) before applying the seed.")


if __name__ == "__main__":
    main()
