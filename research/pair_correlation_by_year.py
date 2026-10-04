#!/usr/bin/env python
"""Per-year correlation drift for every pair in four_sigmas.py.

Splits the 1000-day sample into four 250-day years and reports, for each
configured pair, the log-return correlation and the log-price level
correlation in each year, plus the year-on-year changes.

Return correlation is the primary column: it is what the hedge relies on
day to day. Level correlation is reported alongside because a pair can hold
a high level correlation (co-trending) while its return correlation decays,
and that combination is the classic decaying-pair signature.

Output: analysis_outputs/pair_correlation_by_year/ (csv + markdown).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import four_sigmas as core

OUTPUT_DIR = ROOT / "analysis_outputs" / "pair_correlation_by_year"
YEARS = (("Y1", 1, 250), ("Y2", 251, 500), ("Y3", 501, 750), ("Y4", 751, 1000))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if min(float(a.std()), float(b.std())) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(ROOT / "prices.txt", sep=r"\s+", header=0).to_numpy(float).T
    log_prices = np.log(np.maximum(prices, 1e-12))
    symbol_by_index = {index: symbol for symbol, index in core.I.items()}

    records = []
    for spec in core.PAIR_CFGS:
        y, x = int(spec.y), int(spec.x)
        row = {
            "pair": f"{symbol_by_index[y]}-{symbol_by_index[x]}",
            "y": symbol_by_index[y],
            "x": symbol_by_index[x],
        }
        for label, start, end in YEARS:
            block = log_prices[[y, x], start - 1:end]
            returns = np.diff(block, axis=1)
            row[f"ret_{label}"] = safe_corr(returns[0], returns[1])
            row[f"lvl_{label}"] = safe_corr(block[0], block[1])
        ret = np.array([row[f"ret_{lab}"] for lab, _, _ in YEARS], dtype=float)
        row["d_Y2_Y1"] = ret[1] - ret[0]
        row["d_Y3_Y2"] = ret[2] - ret[1]
        row["d_Y4_Y3"] = ret[3] - ret[2]
        row["d_Y4_Y1"] = ret[3] - ret[0]
        row["ret_mean"] = float(np.nanmean(ret))
        row["ret_min"] = float(np.nanmin(ret))
        row["ret_spread"] = float(np.nanmax(ret) - np.nanmin(ret))
        # Least-squares slope across the four years: the drift direction.
        finite = np.isfinite(ret)
        row["ret_trend"] = (
            float(np.polyfit(np.arange(4)[finite], ret[finite], 1)[0])
            if finite.sum() >= 2 else float("nan")
        )
        row["sign_flips"] = int(np.sum(np.diff(np.sign(ret[finite])) != 0))
        records.append(row)

    frame = pd.DataFrame.from_records(records)
    frame = frame.sort_values("d_Y4_Y1")
    frame.to_csv(OUTPUT_DIR / "pair_correlation_by_year.csv", index=False)

    lines = [
        "# Per-year return correlation for every four_sigmas pair",
        "",
        f"{len(frame)} pairs, four 250-day years "
        "(Y1 1-250, Y2 251-500, Y3 501-750, Y4 751-1000).",
        "",
        "`ret_*` is log-return correlation; `trend` is the least-squares slope "
        "across the four years (per year); `spread` is max-min. Sorted by "
        "Y4-Y1, most-degraded first.",
        "",
        "| Pair | Y1 | Y2 | Y3 | Y4 | Y2-Y1 | Y3-Y2 | Y4-Y3 | **Y4-Y1** | spread | trend | min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in frame.itertuples(index=False):
        lines.append(
            f"| {r.pair} | {r.ret_Y1:.3f} | {r.ret_Y2:.3f} | {r.ret_Y3:.3f} | "
            f"{r.ret_Y4:.3f} | {r.d_Y2_Y1:+.3f} | {r.d_Y3_Y2:+.3f} | "
            f"{r.d_Y4_Y3:+.3f} | **{r.d_Y4_Y1:+.3f}** | {r.ret_spread:.3f} | "
            f"{r.ret_trend:+.3f} | {r.ret_min:.3f} |"
        )

    lines += [
        "",
        "## Level correlation (log prices)",
        "",
        "| Pair | Y1 | Y2 | Y3 | Y4 | Y4-Y1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in frame.itertuples(index=False):
        lines.append(
            f"| {r.pair} | {r.lvl_Y1:.3f} | {r.lvl_Y2:.3f} | {r.lvl_Y3:.3f} | "
            f"{r.lvl_Y4:.3f} | {r.lvl_Y4 - r.lvl_Y1:+.3f} |"
        )
    (OUTPUT_DIR / "pair_correlation_by_year.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    ret_cols = [f"ret_{lab}" for lab, _, _ in YEARS]
    print(f"{len(frame)} pairs. Mean return correlation by year:")
    for lab, _, _ in YEARS:
        print(f"  {lab}: {frame[f'ret_{lab}'].mean():+.4f}")
    print(f"\nY4-Y1 change: mean {frame.d_Y4_Y1.mean():+.4f}  "
          f"median {frame.d_Y4_Y1.median():+.4f}")
    print(f"pairs with LOWER correlation in Y4 than Y1: "
          f"{int((frame.d_Y4_Y1 < 0).sum())}/{len(frame)}")
    print(f"pairs whose correlation changes sign at least once: "
          f"{int((frame.sign_flips > 0).sum())}/{len(frame)}")
    print(f"mean spread (max-min across years): {frame.ret_spread.mean():.4f}")
    print(f"\n10 most degraded (Y4-Y1):")
    print(frame[["pair"] + ret_cols + ["d_Y4_Y1", "ret_spread"]]
          .head(10).round(3).to_string(index=False))
    print(f"\n10 most improved (Y4-Y1):")
    print(frame[["pair"] + ret_cols + ["d_Y4_Y1", "ret_spread"]]
          .tail(10).round(3).to_string(index=False))
    print(f"\nwrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
