#!/usr/bin/env python
"""
Simple pairs mean-reversion trader on the consensus best cointegrated pairs.

Rolling OLS hedge (100d) → z-score entry/exit. Same signal used in teamName.py.
Writes a research backtest (last 250 days, fees/limits matching eval.py).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("cointegration_output/pairs_mean_reversion")

# Consensus shortlist: walk-forward TOP_PAIRS ∩ activity / GH-stable leaders
TRADE_PAIRS = [
    ("AENO", "NWIG"),
    ("EORC", "NGTE"),
    ("SMAH", "NGTE"),
    ("ALGO", "SMAH"),
    ("SMAH", "CTGI"),
    ("HRET", "ULXY"),
]

LOOKBACK = 100
ENTRY_Z = 2.0
EXIT_Z = 0.5
DOLLAR_PER_PAIR = 7_000

DEFAULT_DLR_LIMIT = 10_000
INST0_DLR_LIMIT = 100_000
DEFAULT_COMM = 0.0001
INST0_COMM = 0.00002
NUM_TEST_DAYS = 250


def load_prices(path: str = "prices.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=0)


def ols_mu_beta(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    Z = np.column_stack([np.ones(len(x)), x])
    mu, beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(mu), float(beta)


def causal_z(y: np.ndarray, x: np.ndarray, lookback: int = LOOKBACK) -> tuple[float, float, float]:
    """
    Causal z at the last day: fit hedge on [t-W, t), residual on day t.
    Returns (z, mu, beta); z is nan if insufficient history.
    """
    n = len(y)
    if n < lookback + 1:
        return np.nan, np.nan, np.nan
    y_tr, x_tr = y[-(lookback + 1) : -1], x[-(lookback + 1) : -1]
    if np.std(y_tr) < 1e-12 or np.std(x_tr) < 1e-12:
        return np.nan, np.nan, np.nan
    mu, beta = ols_mu_beta(y_tr, x_tr)
    resid_tr = y_tr - mu - beta * x_tr
    sig = float(resid_tr.std())
    if sig < 1e-12:
        return np.nan, mu, beta
    resid_t = float(y[-1] - mu - beta * x[-1])
    return resid_t / sig, mu, beta


def update_pair_side(prev: int, z: float, entry: float = ENTRY_Z, exit_: float = EXIT_Z) -> int:
    """Hysteresis: +1 long spread, -1 short spread, 0 flat."""
    if np.isnan(z):
        return 0
    if prev == 0:
        if z > entry:
            return -1
        if z < -entry:
            return 1
        return 0
    if prev > 0:
        return 0 if z >= -exit_ else 1
    return 0 if z <= exit_ else -1


def pair_share_targets(
    side: int,
    price_y: float,
    price_x: float,
    beta: float,
    dollar: float = DOLLAR_PER_PAIR,
) -> tuple[int, int]:
    """Dollar-ish long/short: +side shares of y, -side*beta shares of x, scaled to ~dollar on y."""
    if side == 0 or not np.isfinite(beta) or price_y <= 0 or price_x <= 0:
        return 0, 0
    n_y = int(dollar / price_y) * side
    if n_y == 0:
        return 0, 0
    n_x = int(round(-beta * n_y))
    return int(n_y), int(n_x)


def simulate(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = prices.columns.tolist()
    idx = {t: i for i, t in enumerate(tickers)}
    n_inst, n_days = len(tickers), len(prices)
    prc = prices.values.T  # (nInst, nDays)

    dlr_limit = np.full(n_inst, DEFAULT_DLR_LIMIT, dtype=float)
    dlr_limit[0] = INST0_DLR_LIMIT
    comm_rate = np.full(n_inst, DEFAULT_COMM, dtype=float)
    comm_rate[0] = INST0_COMM

    start = n_days - NUM_TEST_DAYS
    pair_sides = {p: 0 for p in TRADE_PAIRS}

    cash = 0.0
    cur_pos = np.zeros(n_inst, dtype=int)
    value = 0.0
    tot_dvol = 0.0
    rows = []
    z_rows = []

    for t in range(start, n_days):
        hist = prc[:, : t + 1]
        cur_px = hist[:, -1]
        desired = np.zeros(n_inst, dtype=float)

        for a, b in TRADE_PAIRS:
            ia, ib = idx[a], idx[b]
            z, mu, beta = causal_z(hist[ia], hist[ib])
            side = update_pair_side(pair_sides[(a, b)], z)
            pair_sides[(a, b)] = side
            ny, nx = pair_share_targets(side, cur_px[ia], cur_px[ib], beta)
            desired[ia] += ny
            desired[ib] += nx
            z_rows.append(
                {
                    "day": t + 1,
                    "pair": f"{a}-{b}",
                    "z": z,
                    "side": side,
                    "beta": beta,
                    "n_y": ny,
                    "n_x": nx,
                }
            )

        pos_limits = (dlr_limit / cur_px).astype(int)
        new_pos = np.clip(np.round(desired), -pos_limits, pos_limits).astype(int)

        delta = new_pos - cur_pos
        dvolumes = cur_px * np.abs(delta)
        comm = float(np.sum(dvolumes * comm_rate))
        cash -= float(cur_px.dot(delta) + comm)
        tot_dvol += float(dvolumes.sum())
        cur_pos = new_pos
        pos_value = float(cur_pos.dot(cur_px))
        today_pl = cash + pos_value - value
        value = cash + pos_value
        rows.append(
            {
                "day": t + 1,
                "pl": today_pl,
                "value": value,
                "dvolume": tot_dvol,
                "n_active_pairs": sum(1 for s in pair_sides.values() if s != 0),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(z_rows)


def plot_results(bt: pd.DataFrame, zdf: pd.DataFrame) -> None:
    cum = bt["pl"].cumsum()
    mu, sig = bt["pl"].mean(), bt["pl"].std()
    sharpe = np.sqrt(250) * mu / sig if sig > 1e-12 else 0.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    axes[0].plot(bt["day"], cum, color="#4C72B0", lw=1.4)
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].set_ylabel("Cumulative PL ($)")
    axes[0].set_title(
        f"Pairs MR ({len(TRADE_PAIRS)} pairs, W={LOOKBACK}, |z|>{ENTRY_Z}/{EXIT_Z}) — "
        f"meanPL={mu:.1f}  annSharpe≈{sharpe:.2f}"
    )

    axes[1].plot(bt["day"], bt["n_active_pairs"], color="#8172B2", lw=1)
    axes[1].set_ylabel("# active pairs")
    axes[1].set_xlabel("Day")

    for pair, g in zdf.groupby("pair"):
        axes[2].plot(g["day"], g["z"], lw=0.9, alpha=0.85, label=pair)
    axes[2].axhline(ENTRY_Z, color="red", ls="--", lw=0.9)
    axes[2].axhline(-ENTRY_Z, color="red", ls="--", lw=0.9)
    axes[2].axhline(EXIT_Z, color="orange", ls=":", lw=0.9)
    axes[2].axhline(-EXIT_Z, color="orange", ls=":", lw=0.9)
    axes[2].axhline(0, color="gray", lw=0.6)
    axes[2].set_ylabel("z-score")
    axes[2].set_xlabel("Day")
    axes[2].set_ylim(-5, 5)
    axes[2].legend(fontsize=7, ncol=3, loc="upper right")

    plt.tight_layout()
    path = OUTPUT_DIR / "pairs_mr_backtest.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    bt, zdf = simulate(prices)
    bt.to_csv(OUTPUT_DIR / "daily_pl.csv", index=False)
    zdf.to_csv(OUTPUT_DIR / "pair_signals.csv", index=False)

    mu, sig = bt["pl"].mean(), bt["pl"].std()
    sharpe = np.sqrt(250) * mu / sig if sig > 1e-12 else 0.0
    score = mu if mu <= 0 or sig < 1e-10 else mu * (sharpe**2 / (sharpe**2 + 1.0))
    print(f"Pairs: {[f'{a}-{b}' for a, b in TRADE_PAIRS]}")
    print(f"mean(PL): {mu:.1f}")
    print(f"StdDev(PL): {sig:.2f}")
    print(f"annSharpe: {sharpe:.2f}")
    print(f"Score~: {score:.2f}")
    print(f"Final value: {bt['value'].iloc[-1]:.2f}")
    plot_results(bt, zdf)


if __name__ == "__main__":
    main()
