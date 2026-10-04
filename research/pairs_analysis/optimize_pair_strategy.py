"""Per-pair PairSegFilter optimisation + staggered walk-forward MR strategy.

Optimises filter + z-score mean-reversion params on 500 in-sample days,
then scores on out-of-sample windows staggered by offset.

Uses the Algothon score: mu * sr^2/(sr^2+1) with 1bp commission and $10k legs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_beta import PairSegConfig, run_pair_seg
from prices import load_prices

OUT = Path("cusum_output/pair_strategy_wf")
IS_LEN = 500
OS_MIN = 50
OFFSETS = (0, 50, 100, 150, 200)
LEG = 10_000.0
COMM = 0.0001  # 1bp

PAIR_I = np.array([10, 1, 13, 8, 25, 49, 7, 31, 36, 18, 33, 35, 26], dtype=int)
PAIR_J = np.array([46, 20, 45, 27, 37, 50, 40, 43, 41, 35, 26, 28, 32], dtype=int)
PAIRS = tuple(zip(PAIR_I.tolist(), PAIR_J.tolist()))


@dataclass(frozen=True)
class TradeParams:
    entry_z: float = 1.75
    exit_z: float = 0.50
    stop_z: float = 4.00
    z_ewma: float = 0.94  # spread z scale
    trade_while_adapting: bool = False


def algothon_score(pll: np.ndarray, param: float = 1.0) -> tuple[float, float, float, float]:
    pll = np.asarray(pll, dtype=float)
    if pll.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mu = float(np.mean(pll))
    sigma = float(np.std(pll))
    if sigma < 1e-10:
        sharpe = 0.0
        sc = mu
    else:
        sharpe = float(np.sqrt(250.0) * mu / sigma)
        if mu <= 0.0:
            sc = mu
        else:
            frac = sharpe**2 / (sharpe**2 + param**2)
            sc = mu * frac
    return sc, mu, sigma, sharpe


def _z_series(spread: np.ndarray, lam: float) -> np.ndarray:
    """Causal EWMA mean/var z-score of the filter spread."""
    T = spread.size
    z = np.zeros(T)
    mean = float(spread[0])
    var = max(float(spread[0] * spread[0]) * 1e-4, 1e-8)
    for t in range(T):
        s = float(spread[t])
        if t == 0:
            z[t] = 0.0
            continue
        mean = lam * mean + (1.0 - lam) * s
        var = lam * var + (1.0 - lam) * (s - mean) ** 2
        z[t] = (s - mean) / np.sqrt(max(var, 1e-12))
    return z


def simulate_pair(
    y: np.ndarray,
    x: np.ndarray,
    filt_cfg: PairSegConfig,
    trade: TradeParams,
    *,
    score_start: int,
    score_end: int,
) -> dict:
    """Run filter on [0, score_end); trade/score on [score_start, score_end).

    Critical: hedge beta is FROZEN at entry.  Live filter beta is only used to
    open/resize after a flat.  CUSUM cuts flatten the book — a regime change
    invalidates the open mean-reversion trade and the residual jumps.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    end = min(score_end, y.size)
    start = max(0, score_start)
    if end - start < OS_MIN:
        return {
            "score": float("nan"),
            "mu": float("nan"),
            "sigma": float("nan"),
            "sharpe": float("nan"),
            "n_days": 0,
            "n_trades": 0,
            "pll": np.array([]),
        }

    res = run_pair_seg(y[:end], x[:end], filt_cfg)

    # Trading residual uses frozen entry beta — NOT the jumping filter residual.
    # z is still computed on a causal EWMA of the *held* residual path.
    pos_y = 0
    pos_x = 0
    side = 0  # +1 long spread, -1 short
    hedge_beta = 0.0
    cash = 0.0
    value = 0.0
    pll: list[float] = []
    n_trades = 0

    # EWMA z of the economic residual (y - hedge_beta * x) while flat uses live beta
    mean = 0.0
    var = 1e-4
    lam = trade.z_ewma
    z_ready = False

    for t in range(end):
        py, px = float(y[t]), float(x[t])
        live_beta = float(res.beta[t])
        adapting = bool(res.adapting[t])
        cut = bool(res.cut[t])

        # Residual for signal: held hedge if in trade, else live filter beta
        b_sig = hedge_beta if side != 0 else live_beta
        resid = py - b_sig * px
        if not z_ready:
            mean = resid
            var = max(resid * resid * 1e-4, 1e-8)
            z_ready = True
            zt = 0.0
        else:
            mean = lam * mean + (1.0 - lam) * resid
            var = lam * var + (1.0 - lam) * (resid - mean) ** 2
            zt = (resid - mean) / np.sqrt(max(var, 1e-12))

        new_side = side
        new_hedge = hedge_beta

        if t < end - 1 and t >= max(0, start - 1):
            # Regime snap: flatten — residual is discontinuous under new beta
            if cut and side != 0:
                new_side = 0
                new_hedge = 0.0
            else:
                can_trade = trade.trade_while_adapting or (not adapting)
                if side == 0 and can_trade:
                    if zt >= trade.entry_z:
                        new_side = -1
                        new_hedge = live_beta
                    elif zt <= -trade.entry_z:
                        new_side = 1
                        new_hedge = live_beta
                elif side != 0:
                    if abs(zt) <= trade.exit_z or abs(zt) >= trade.stop_z:
                        new_side = 0
                        new_hedge = 0.0
                    # no flip through live beta — exit first, re-enter next bar

        if new_side == 0:
            tgt_y, tgt_x = 0, 0
        else:
            b = new_hedge if new_hedge != 0.0 else live_beta
            # Keep hedge proportional under $LEG/leg caps:
            # |N_y| <= LEG, |N_x| = |β|·|N_y| <= LEG  =>  |N_y| = LEG / max(1, |β|)
            n_y = new_side * LEG / max(1.0, abs(b))
            n_x = -new_side * b * (LEG / max(1.0, abs(b)))
            tgt_y = int(n_y / max(py, 1e-8))
            tgt_x = int(n_x / max(px, 1e-8))

        if t < end - 1:
            dy = tgt_y - pos_y
            dx = tgt_x - pos_x
            dvol = abs(dy) * py + abs(dx) * px
            fee = dvol * COMM
            cash -= dy * py + dx * px + fee
            if side == 0 and new_side != 0:
                n_trades += 1
            pos_y, pos_x = tgt_y, tgt_x
            side = new_side
            hedge_beta = new_hedge

        pos_val = pos_y * py + pos_x * px
        today_pl = cash + pos_val - value
        value = cash + pos_val
        if t >= start:
            pll.append(today_pl)

    pll_a = np.asarray(pll, dtype=float)
    sc, mu, sigma, sharpe = algothon_score(pll_a)
    return {
        "score": sc,
        "mu": mu,
        "sigma": sigma,
        "sharpe": sharpe,
        "n_days": int(pll_a.size),
        "n_trades": n_trades,
        "pll": pll_a,
    }


def filter_grid() -> list[PairSegConfig]:
    base = PairSegConfig()
    out: list[PairSegConfig] = []
    for h, k, mseg, over in product(
        (6.0, 10.0, 14.0),
        (0.45, 0.65),
        (30, 45),
        (1.15,),
    ):
        out.append(
            replace(
                base,
                h=h,
                cusum_k=k,
                min_seg_len=mseg,
                snap_overshoot=over,
                target_seg=max(30, mseg),
            )
        )
    return out


def trade_grid() -> list[TradeParams]:
    out: list[TradeParams] = []
    for entry, exit_z, stop, zlam in product(
        (1.25, 1.75, 2.25),
        (0.35, 0.60),
        (3.75,),
        (0.94,),
    ):
        if exit_z >= entry:
            continue
        out.append(
            TradeParams(entry_z=entry, exit_z=exit_z, stop_z=stop, z_ewma=zlam)
        )
    return out


def optimise_on_train(
    y: np.ndarray,
    x: np.ndarray,
    train_end: int,
) -> tuple[PairSegConfig, TradeParams, float]:
    """Grid search on days [0, train_end). Returns best configs + IS score."""
    y_tr, x_tr = y[:train_end], x[:train_end]
    # score the second half of train to reduce pure in-sample curve-fit a bit
    score_start = max(IS_LEN // 5, train_end - 250)
    best_sc = -1e18
    best_f = PairSegConfig()
    best_t = TradeParams()

    # Stage 1: filter with default trade
    default_trade = TradeParams()
    filt_scores: list[tuple[float, PairSegConfig]] = []
    for fcfg in filter_grid():
        r = simulate_pair(
            y_tr, x_tr, fcfg, default_trade, score_start=score_start, score_end=train_end
        )
        sc = r["score"]
        if np.isfinite(sc):
            filt_scores.append((sc, fcfg))
    filt_scores.sort(key=lambda z: z[0], reverse=True)
    top_filters = [f for _, f in filt_scores[:8]] or [PairSegConfig()]

    # Stage 2: trade params on top filters
    for fcfg in top_filters:
        for tcfg in trade_grid():
            r = simulate_pair(
                y_tr, x_tr, fcfg, tcfg, score_start=score_start, score_end=train_end
            )
            sc = r["score"]
            if np.isfinite(sc) and sc > best_sc:
                best_sc = sc
                best_f, best_t = fcfg, tcfg

    return best_f, best_t, float(best_sc)


def run_walk_forward(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = list(prices.columns)
    T = len(prices)
    rows: list[dict] = []
    best_rows: list[dict] = []

    for yi, xj in PAIRS:
        y_t, x_t = tickers[yi], tickers[xj]
        y = prices[y_t].to_numpy(dtype=float)
        x = prices[x_t].to_numpy(dtype=float)
        print(f"\n=== {y_t}/{x_t} (idx {yi}/{xj}) ===")

        oos_scores: list[float] = []
        for off in OFFSETS:
            tr0, tr1 = off, off + IS_LEN
            te0, te1 = tr1, min(tr1 + 200, T)
            if tr1 > T or te1 - te0 < OS_MIN:
                continue

            # Slice so day 0 of arrays = global day `off`
            y_w = y[off:te1]
            x_w = x[off:te1]
            train_end = IS_LEN  # relative
            test_start = IS_LEN
            test_end = te1 - off

            fcfg, tcfg, is_sc = optimise_on_train(y_w, x_w, train_end)
            oos = simulate_pair(
                y_w,
                x_w,
                fcfg,
                tcfg,
                score_start=test_start,
                score_end=test_end,
            )
            oos_sc = oos["score"]
            oos_scores.append(oos_sc)
            row = {
                "pair": f"{y_t}/{x_t}",
                "y_idx": yi,
                "x_idx": xj,
                "offset": off,
                "train": f"{tr0}:{tr1}",
                "test": f"{te0}:{te1}",
                "is_score": is_sc,
                "oos_score": oos_sc,
                "oos_mu": oos["mu"],
                "oos_sigma": oos["sigma"],
                "oos_sharpe": oos["sharpe"],
                "oos_days": oos["n_days"],
                "oos_trades": oos["n_trades"],
                "h": fcfg.h,
                "cusum_k": fcfg.cusum_k,
                "min_seg_len": fcfg.min_seg_len,
                "snap_overshoot": fcfg.snap_overshoot,
                "entry_z": tcfg.entry_z,
                "exit_z": tcfg.exit_z,
                "stop_z": tcfg.stop_z,
                "z_ewma": tcfg.z_ewma,
            }
            rows.append(row)
            print(
                f"  off={off:3d}  IS={is_sc:7.2f}  OOS={oos_sc:7.2f}  "
                f"Sharpe={oos['sharpe']:5.2f}  trades={oos['n_trades']:3d}  "
                f"h={fcfg.h} k={fcfg.cusum_k} entry={tcfg.entry_z}"
            )

        finite = [s for s in oos_scores if np.isfinite(s)]
        best_rows.append(
            {
                "pair": f"{y_t}/{x_t}",
                "y_idx": yi,
                "x_idx": xj,
                "n_windows": len(finite),
                "oos_score_mean": float(np.mean(finite)) if finite else float("nan"),
                "oos_score_median": float(np.median(finite)) if finite else float("nan"),
                "oos_score_std": float(np.std(finite)) if finite else float("nan"),
                "oos_score_min": float(np.min(finite)) if finite else float("nan"),
                "oos_score_max": float(np.max(finite)) if finite else float("nan"),
                "frac_positive": float(np.mean(np.array(finite) > 0)) if finite else 0.0,
            }
        )
        print(
            f"  >> mean OOS score={best_rows[-1]['oos_score_mean']:.2f}  "
            f"median={best_rows[-1]['oos_score_median']:.2f}  "
            f"pos_windows={best_rows[-1]['frac_positive']:.0%}"
        )

    detail = pd.DataFrame(rows)
    summary = pd.DataFrame(best_rows).sort_values(
        "oos_score_mean", ascending=False
    ).reset_index(drop=True)
    return detail, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prices = load_prices("prices.txt")
    detail, summary = run_walk_forward(prices)

    detail_path = OUT / "wf_detail.csv"
    summary_path = OUT / "wf_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    lines = [
        "# PairSegFilter walk-forward pair strategy",
        "",
        f"- In-sample length: `{IS_LEN}` days",
        f"- Offsets: `{list(OFFSETS)}`",
        f"- OOS length: up to 200 days after each train window",
        f"- Leg notional: `${LEG:.0f}` / side, commission `{COMM}`",
        "",
        "## Mean OOS Algothon score by pair",
        "",
        "| Pair | Windows | Mean score | Median | Std | Min | Max | % positive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['pair']} | {int(r['n_windows'])} | {r['oos_score_mean']:.2f} | "
            f"{r['oos_score_median']:.2f} | {r['oos_score_std']:.2f} | "
            f"{r['oos_score_min']:.2f} | {r['oos_score_max']:.2f} | "
            f"{100 * r['frac_positive']:.0f}% |"
        )
    lines.extend(
        [
            "",
            f"Detail: `{detail_path.name}`",
            f"Summary: `{summary_path.name}`",
            "",
        ]
    )
    (OUT / "wf_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n==== SUMMARY (mean OOS score) ====")
    print(
        summary[
            [
                "pair",
                "n_windows",
                "oos_score_mean",
                "oos_score_median",
                "frac_positive",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote {OUT.resolve()}")


if __name__ == "__main__":
    main()
