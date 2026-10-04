"""Multi-pair dashboard for the lean PairSegFilter (sticky + burst adapt)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from adaptive_beta import PairSegConfig, run_pair_seg
from prices import extract_pair, load_prices

OUT = Path("cusum_output/multi_pair_dashboard")
W_RAW = 100

# Mix of trading-book pairs + prior research focus pairs
PAIRS = [
    ("SMAH", "NGTE"),
    ("SMAH", "ILVX"),
    ("EORC", "NGTE"),
    ("AENO", "NWIG"),
    ("HUXZ", "ACAC"),
    ("SMAH", "CTGI"),
    ("ELLT", "ILVX"),
    ("GARI", "IHOZ"),
    ("HRET", "MMBT"),
    ("ALGO", "SMAH"),
    ("AENO", "RRES"),
    ("HRET", "FCSG"),
]


def rolling_through_origin(y: np.ndarray, x: np.ndarray, w: int = W_RAW) -> np.ndarray:
    n = len(y)
    out = np.full(n, np.nan)
    for i in range(w, n):
        xi, yi = x[i - w : i], y[i - w : i]
        d = float(xi @ xi)
        if d > 1e-12:
            out[i] = float(xi @ yi) / d
    return out


def run_pair(y: np.ndarray, x: np.ndarray) -> dict:
    cfg = PairSegConfig()
    res = run_pair_seg(y, x, cfg)
    raw = rolling_through_origin(y, x)
    snaps = np.flatnonzero(res.cut)
    stable = ~res.adapting
    return {
        "y": y,
        "x": x,
        "beta": res.beta,
        "raw": raw,
        "spread": res.spread,
        "adapting": res.adapting.astype(float),
        "alarms": snaps,
        "adapt_frac": float(res.adapting.mean()),
        "n_snaps": int(snaps.size),
        "spread_max": float(np.max(np.abs(res.spread))),
        "beta_mad_stable": float(
            np.mean(np.abs(np.diff(res.beta[stable])))
            if np.any(stable[:-1])
            else np.nan
        ),
        "final_h": float(res.h[-1]),
        "ok": True,
        "error": None,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prices = load_prices("prices.txt")

    results: list[tuple[str, str, dict]] = []
    for y_t, x_t in PAIRS:
        try:
            y, x = extract_pair(prices, y_t, x_t)
            info = run_pair(y, x)
            results.append((y_t, x_t, info))
            print(
                f"OK  {y_t}/{x_t}: snaps={info['n_snaps']} "
                f"adapt={info['adapt_frac']:.2f} |spread|_max={info['spread_max']:.1f} "
                f"h={info['final_h']:.2f}"
            )
        except Exception as exc:  # noqa: BLE001 — dashboard should skip bad pairs
            print(f"SKIP {y_t}/{x_t}: {exc}")
            results.append(
                (
                    y_t,
                    x_t,
                    {
                        "ok": False,
                        "error": str(exc),
                        "y": None,
                        "x": None,
                    },
                )
            )

    ok_rows = [(a, b, r) for a, b, r in results if r.get("ok")]
    n = len(ok_rows)
    if n == 0:
        raise SystemExit("No pairs succeeded")

    fig = plt.figure(figsize=(16, 2.15 * n + 0.8), constrained_layout=False)
    gs = GridSpec(
        n,
        3,
        figure=fig,
        hspace=0.55,
        wspace=0.28,
        left=0.06,
        right=0.98,
        top=0.94,
        bottom=0.04,
    )

    fig.suptitle(
        "PairSegFilter dashboard  ·  sticky β  ·  CUSUM snap + 8-bar burst",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )

    for i, (y_t, x_t, r) in enumerate(ok_rows):
        y, x = r["y"], r["x"]
        t = np.arange(len(y))
        label = f"{y_t}/{x_t}"

        ax = fig.add_subplot(gs[i, 0])
        ax.plot(t, y / y[0], color="#4C72B0", lw=0.9, label=y_t)
        ax.plot(t, x / x[0], color="#DD8452", lw=0.9, label=x_t)
        for idx in r["alarms"]:
            ax.axvline(idx, color="#c44e52", ls="--", lw=0.6, alpha=0.55)
        ax.set_ylabel(
            label, fontsize=8, fontweight="bold", rotation=0, labelpad=28, va="center"
        )
        if i == 0:
            ax.set_title("Price / day1", fontsize=10)
            ax.legend(fontsize=6, loc="upper left", framealpha=0.7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)
        if i < n - 1:
            ax.set_xticklabels([])

        ax = fig.add_subplot(gs[i, 1])
        ax.plot(t, r["raw"], color="#333333", lw=1.1, alpha=0.85, label=f"raw {W_RAW}d")
        ax.plot(t, r["beta"], color="#1f4e79", lw=1.15, label="PairSeg β")
        ax.fill_between(t, 0, r["adapting"], step="pre", color="#f4a261", alpha=0.35)
        for idx in r["alarms"]:
            ax.axvline(idx, color="#c44e52", ls="--", lw=0.6, alpha=0.55)
        finite_raw = r["raw"][np.isfinite(r["raw"])]
        lo = min(float(np.nanmin(r["beta"])), float(np.nanmin(finite_raw))) - 0.15
        hi = max(float(np.nanmax(r["beta"])), float(np.nanmax(finite_raw))) + 0.15
        ax.set_ylim(lo, hi)
        if i == 0:
            ax.set_title("Beta (sticky → burst snap)", fontsize=10)
            ax.legend(fontsize=6, loc="upper left", framealpha=0.7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)
        if i < n - 1:
            ax.set_xticklabels([])
        ax.text(
            0.99,
            0.05,
            f"snaps={r['n_snaps']}  adapt={r['adapt_frac']:.0%}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color="#333",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", alpha=0.8),
        )

        ax = fig.add_subplot(gs[i, 2])
        ax.plot(t, r["spread"], color="#2ca02c", lw=0.75)
        ax.axhline(0.0, color="gray", lw=0.5)
        for idx in r["alarms"]:
            ax.axvline(idx, color="#c44e52", ls="--", lw=0.6, alpha=0.55)
        if i == 0:
            ax.set_title("Spread = y − β·x", fontsize=10)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)
        if i < n - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("day index", fontsize=8)
        ax.text(
            0.99,
            0.05,
            f"|s|_max={r['spread_max']:.1f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color="#333",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", alpha=0.8),
        )

    path = OUT / "multi_pair_beta_dashboard.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\nDashboard: {path.resolve()}")

    lines = [
        "# Multi-pair PairSegFilter dashboard",
        "",
        "Filter: `adaptive_beta.PairSegFilter` (sticky through-origin + CUSUM "
        "snap + 8-bar burst).",
        "",
        "| Pair | Snaps | Adapt % | max|spread| | Stable β mad | final h |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for y_t, x_t, r in ok_rows:
        lines.append(
            f"| {y_t}/{x_t} | {r['n_snaps']} | {100 * r['adapt_frac']:.1f}% | "
            f"{r['spread_max']:.2f} | {r['beta_mad_stable']:.4f} | {r['final_h']:.2f} |"
        )
    lines.extend(["", f"Plot: `{path.name}`", ""])
    (OUT / "multi_pair_dashboard_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
