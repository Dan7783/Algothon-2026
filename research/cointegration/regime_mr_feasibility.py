#!/usr/bin/env python
"""
Feasibility: if we knew the active regime, is within-regime spread MR tradable?

Upper-bound test using Gregory-Hansen break dates (ex-post = look-ahead).
If EVEN this fails, real-time regime ID won't save the signal.
If it works, the bottleneck is causal regime detection, not MR itself.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from research.cointegration.cointegration_output.gregory_hansen_lc import gregory_hansen, hansen_lc, TOP_PAIRS, load_prices

OUTPUT_DIR = Path("cointegration_output/regime_mr_feasibility")
ENTRY_Z = 1.5
EXIT_Z = 0.25
MIN_REGIME_LEN = 80  # need enough days to estimate + trade
TRAIN_FRAC = 0.5  # within each regime: first half fit, second half trade OOS


def hedge_ratio(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    Z = np.column_stack([np.ones(len(x)), x])
    mu, beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(mu), float(beta)


def half_life(spread: np.ndarray) -> float:
    if len(spread) < 15 or np.std(spread) < 1e-12:
        return np.nan
    lag, cur = spread[:-1], spread[1:]
    Z = np.column_stack([np.ones(len(lag)), lag])
    _, rho = np.linalg.lstsq(Z, cur, rcond=None)[0]
    if not (0 < rho < 1):
        return np.nan
    return float(-np.log(2) / np.log(rho))


def mr_pnl(y: np.ndarray, x: np.ndarray, mu: float, beta: float, sigma: float) -> dict:
    """OOS z-score MR using fixed (mu, beta, sigma) from train."""
    if sigma < 1e-12:
        return {"pnl": 0.0, "n_trades": 0, "hit": 0.0, "avg_hold": 0.0}
    spread = y - mu - beta * x
    z = (spread - 0.0) / sigma  # train demeaned residual; mu already in spread def
    # actually spread already has mu removed if y - mu - beta x; center at 0
    z = spread / sigma

    pos = 0.0
    entry_s = 0.0
    entry_i = 0
    pnls = []
    holds = []

    for i in range(1, len(spread)):
        if pos == 0:
            if z[i - 1] > ENTRY_Z:
                pos, entry_s, entry_i = -1.0, spread[i - 1], i - 1
            elif z[i - 1] < -ENTRY_Z:
                pos, entry_s, entry_i = 1.0, spread[i - 1], i - 1
        else:
            if (pos > 0 and z[i - 1] >= -EXIT_Z) or (pos < 0 and z[i - 1] <= EXIT_Z):
                trade_pnl = pos * (spread[i - 1] - entry_s)
                pnls.append(trade_pnl)
                holds.append(i - 1 - entry_i)
                pos = 0.0

    if pos != 0:
        pnls.append(pos * (spread[-1] - entry_s))
        holds.append(len(spread) - 1 - entry_i)

    if not pnls:
        return {"pnl": 0.0, "n_trades": 0, "hit": 0.0, "avg_hold": 0.0}
    arr = np.array(pnls)
    return {
        "pnl": float(arr.sum()),
        "n_trades": int(len(arr)),
        "hit": float((arr > 0).mean()),
        "avg_hold": float(np.mean(holds)),
        "avg_trade": float(arr.mean()),
    }


def evaluate_segment(y: np.ndarray, x: np.ndarray, label: str) -> dict:
    n = len(y)
    if n < MIN_REGIME_LEN:
        return None
    split = max(MIN_REGIME_LEN // 2, int(n * TRAIN_FRAC))
    if n - split < 30:
        return None

    y_tr, x_tr = y[:split], x[:split]
    y_te, x_te = y[split:], x[split:]

    mu, beta = hedge_ratio(y_tr, x_tr)
    sp_tr = y_tr - mu - beta * x_tr
    sigma = float(sp_tr.std())
    hl = half_life(sp_tr)

    _, p_tr, _ = coint(y_tr, x_tr)
    adf_p = adfuller(sp_tr, autolag="AIC")[1]

    trade = mr_pnl(y_te, x_te, mu, beta, sigma)
    return {
        "segment": label,
        "n_days": n,
        "train_days": split,
        "test_days": n - split,
        "train_coint_p": p_tr,
        "train_adf_p": adf_p,
        "half_life": hl,
        "sigma": sigma,
        **trade,
    }


def run_pair(prices: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    y = prices[a].values.astype(float)
    x = prices[b].values.astype(float)
    gh = gregory_hansen(y, x, model="C/S")
    br = gh["break_idx"]
    rows = []

    # Full sample (no regime knowledge)
    full = evaluate_segment(y, x, "full_sample")
    if full:
        full.update({"pair": f"{a}-{b}", "break_day": br + 1 if br is not None else None, "gh_sig": gh["reject_5"]})
        rows.append(full)

    if br is not None and br >= MIN_REGIME_LEN and len(y) - br >= MIN_REGIME_LEN:
        pre = evaluate_segment(y[:br], x[:br], "regime_pre")
        post = evaluate_segment(y[br:], x[br:], "regime_post")
        for seg in (pre, post):
            if seg:
                seg.update(
                    {
                        "pair": f"{a}-{b}",
                        "break_day": br + 1,
                        "gh_sig": gh["reject_5"],
                    }
                )
                rows.append(seg)

    # Oracle: trade only in the regime where train cointegration is stronger (lower p)
    # Still uses ex-post break; within-regime OOS only
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    # Aggregate hit rate / pnl by segment type
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    order = ["full_sample", "regime_pre", "regime_post"]
    labels = ["Full sample\n(no regime ID)", "Pre-break\nregime (OOS)", "Post-break\nregime (OOS)"]
    hit = [df.loc[df["segment"] == s, "hit"].mean() * 100 if (df["segment"] == s).any() else 0 for s in order]
    pnl = [df.loc[df["segment"] == s, "pnl"].mean() if (df["segment"] == s).any() else 0 for s in order]
    ntr = [df.loc[df["segment"] == s, "n_trades"].mean() if (df["segment"] == s).any() else 0 for s in order]

    axes[0].bar(labels, hit, color=["#999999", "#4C72B0", "#55A868"])
    axes[0].axhline(50, color="red", ls="--", lw=1)
    axes[0].set_ylabel("Mean trade hit rate (%)")
    axes[0].set_title("OOS MR win rate")
    for i, v in enumerate(hit):
        axes[0].text(i, v + 1, f"{v:.0f}%", ha="center")

    axes[1].bar(labels, pnl, color=["#999999", "#4C72B0", "#55A868"])
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_ylabel("Mean total OOS spread PnL")
    axes[1].set_title("OOS PnL (spread units)")

    axes[2].bar(labels, ntr, color=["#999999", "#4C72B0", "#55A868"])
    axes[2].set_ylabel("Mean # trades in test half")
    axes[2].set_title("Activity")

    fig.suptitle(
        "Optimistic upper bound: ex-post GH break known\n"
        f"(within-regime train/test split, entry |z|>{ENTRY_Z})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "regime_vs_full_mr.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_pair_best(df: pd.DataFrame, out_dir: Path) -> None:
    """Per pair: full vs best within-regime hit rate."""
    pairs = df["pair"].unique()
    rows = []
    for p in pairs:
        sub = df[df["pair"] == p]
        full = sub[sub["segment"] == "full_sample"]
        reg = sub[sub["segment"].isin(["regime_pre", "regime_post"])]
        if full.empty or reg.empty:
            continue
        rows.append(
            {
                "pair": p,
                "full_hit": full["hit"].iloc[0] * 100,
                "best_regime_hit": reg["hit"].max() * 100,
                "full_pnl": full["pnl"].iloc[0],
                "best_regime_pnl": reg.loc[reg["hit"].idxmax(), "pnl"],
            }
        )
    if not rows:
        return
    cmp = pd.DataFrame(rows).sort_values("best_regime_hit", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(cmp))))
    y = np.arange(len(cmp))
    ax.barh(y - 0.2, cmp["full_hit"], height=0.4, label="Full sample", color="#999999")
    ax.barh(y + 0.2, cmp["best_regime_hit"], height=0.4, label="Best regime (oracle break)", color="#55A868")
    ax.axvline(50, color="red", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(cmp["pair"])
    ax.set_xlabel("OOS hit rate (%)")
    ax.set_title("Does knowing the regime help MR? (ex-post break = upper bound)")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "per_pair_regime_lift.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    print(f"Loaded {len(prices)} days")
    print("Optimistic test: GH break known ex-post; MR OOS within each regime\n")

    all_rows = []
    for a, b in TOP_PAIRS:
        part = run_pair(prices, a, b)
        if part is None or part.empty:
            continue
        all_rows.append(part)
        for _, r in part.iterrows():
            print(
                f"{r['pair']:12s} {r['segment']:12s}  "
                f"hit={r['hit']*100:5.1f}%  trades={r['n_trades']:2d}  "
                f"pnl={r['pnl']:8.1f}  HL={r['half_life'] if np.isfinite(r['half_life']) else float('nan'):5.1f}  "
                f"coint_p={r['train_coint_p']:.3f}"
            )

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(OUTPUT_DIR / "regime_mr_results.csv", index=False)
    plot_comparison(df, OUTPUT_DIR)
    plot_pair_best(df, OUTPUT_DIR)

    print("\n=== Aggregate ===")
    for seg in ["full_sample", "regime_pre", "regime_post"]:
        sub = df[df["segment"] == seg]
        if sub.empty:
            continue
        print(
            f"{seg:12s}: n={len(sub):2d}  mean_hit={sub['hit'].mean()*100:.1f}%  "
            f"median_hit={sub['hit'].median()*100:.1f}%  "
            f"mean_pnl={sub['pnl'].mean():.1f}  "
            f"frac_pnl>0={(sub['pnl']>0).mean()*100:.0f}%"
        )

    # Best regime per pair vs full
    lifts = []
    for p in df["pair"].unique():
        sub = df[df["pair"] == p]
        full = sub[sub["segment"] == "full_sample"]
        reg = sub[sub["segment"].isin(["regime_pre", "regime_post"])]
        if full.empty or reg.empty:
            continue
        lifts.append(reg["hit"].max() - full["hit"].iloc[0])
    if lifts:
        print(f"\nMean hit-rate lift (best regime - full): {np.mean(lifts)*100:+.1f} pp")
        print(f"Pairs improved by regime split: {np.mean(np.array(lifts)>0)*100:.0f}%")

    print("\n=== Feasibility verdict helpers ===")
    print("If oracle (ex-post) regimes do NOT improve OOS MR -> not feasible even with perfect regime ID.")
    print("If oracle regimes DO improve but lift is modest -> feasible only with accurate causal regime detector.")
    print("Real-time GH break search needs expanding window; lag will eat part of the edge.")


if __name__ == "__main__":
    main()
