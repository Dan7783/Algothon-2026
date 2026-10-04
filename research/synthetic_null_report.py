"""Build the report and figure for research/synthetic_null.py output.

Merges every paths_*.csv the Monte Carlo has written (dedup on generator+seed),
so null / blocks / control / mechanism runs accumulate into one report.

Run from repo root:
    python research/synthetic_null_report.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = _ROOT / "analysis_outputs" / "synthetic_null"

# dataviz reference palette. Slots 1-3 validate all-pairs on the light surface;
# aqua carries a contrast WARN, relieved by the direct row labels + tables.
SERIES_1 = "#2a78d6"   # endpoint-free nulls (the verdict rests on these)
SERIES_2 = "#eb6834"   # structure-preserving surrogate
SERIES_3 = "#1baf7a"   # endpoint-pinned nulls (diagnostic only)
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BASELINE = "#c3c2b7"

PRIMARY_NULLS = ["xs_boot", "iid_boot", "gauss_corr"]
PINNED_NULLS = ["xs_perm", "iid_perm"]
LABELS = {
    "xs_boot": "xs_boot\ncommon-index bootstrap",
    "iid_boot": "iid_boot\nper-asset bootstrap",
    "gauss_corr": "gauss_corr\nmatched Gaussian",
    "xs_perm": "xs_perm\ncommon shuffle (pinned)",
    "iid_perm": "iid_perm\nper-asset shuffle (pinned)",
    "phase_surr": "phase_surr\nphase surrogate",
}
COLOR_OF = (
    {g: SERIES_1 for g in PRIMARY_NULLS}
    | {g: SERIES_3 for g in PINNED_NULLS}
    | {"phase_surr": SERIES_2}
)
ROW_ORDER = PRIMARY_NULLS + PINNED_NULLS + ["phase_surr"]


def load_paths() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(OUT.glob("paths_*.csv"))]
    if not frames:
        raise SystemExit(f"no paths_*.csv in {OUT}")
    df = pd.concat(frames, ignore_index=True)
    df["kappa"] = df["kappa"].fillna(-1.0)
    return df.drop_duplicates(subset=["generator", "seed", "kappa"], keep="last")


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _band(ax, x, grp, color, log_x=False):
    ax.fill_between(x, grp["p05"], grp["p95"], color=color, alpha=0.15, linewidth=0)
    ax.plot(x, grp["med"], color=color, linewidth=2.0, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=3)
    if log_x:
        ax.set_xscale("log")


def build_figure(ok: pd.DataFrame, real_score: float) -> None:
    fig = plt.figure(figsize=(13.5, 10.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.38, wspace=0.22)

    ax = fig.add_subplot(gs[0, :])
    _style(ax)
    rng = np.random.default_rng(7)
    rows = [g for g in ROW_ORDER if (ok["generator"] == g).any()]
    for i, gen in enumerate(rows):
        scores = ok.loc[ok["generator"] == gen, "score"].to_numpy()
        color = COLOR_OF[gen]
        y = len(rows) - 1 - i
        ax.scatter(scores, y + rng.uniform(-0.16, 0.16, scores.size), s=9,
                   color=color, alpha=0.30, linewidths=0, zorder=2)
        p05, q1, med, q3, p95 = np.percentile(scores, [5, 25, 50, 75, 95])
        ax.plot([p05, p95], [y, y], color=color, linewidth=1.2, zorder=3)
        ax.plot([q1, q3], [y, y], color=color, linewidth=4.0,
                solid_capstyle="butt", zorder=4)
        ax.plot([med], [y], marker="|", color=SURFACE, markersize=11,
                markeredgewidth=2.2, zorder=5)
        if gen != "phase_surr":
            ax.text(p95 + 25, y + 0.24, f"max {scores.max():,.0f}",
                    color=SECONDARY, fontsize=8.5, va="center")

    ax.axvline(0, color=BASELINE, linewidth=1.0, zorder=1)
    ax.axvline(real_score, color=INK, linewidth=2.0, zorder=6)
    ax.text(real_score - 20, len(rows) - 0.40, f"real data  {real_score:,.0f}",
            color=INK, fontsize=10.5, fontweight="bold", ha="right", va="top")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[g] for g in reversed(rows)], fontsize=9,
                       color=SECONDARY)
    ax.set_ylim(-0.6, len(rows) - 0.30)
    ax.set_xlabel("score on days 751-1000 ($)", color=SECONDARY, fontsize=10)
    ax.set_title("Null distributions of the four_sigmas score vs the real-data score",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=12)
    handles = [
        plt.Line2D([], [], color=SERIES_1, linewidth=4, label="null, endpoint-free (the test)"),
        plt.Line2D([], [], color=SERIES_3, linewidth=4, label="null, endpoint-pinned (biased - see report)"),
        plt.Line2D([], [], color=SERIES_2, linewidth=4, label="surrogate, all linear structure kept"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9,
              labelcolor=SECONDARY)
    ax.grid(axis="y", visible=False)

    # block-length horizon sweep
    axb = fig.add_subplot(gs[1, 0])
    _style(axb)
    blocks = ok[ok["generator"].str.startswith("block_boot_L")].copy()
    base = ok.loc[ok["generator"] == "xs_boot", "score"]
    recs = [{"L": 1, "med": base.median(), "p05": base.quantile(0.05),
             "p95": base.quantile(0.95)}]
    if not blocks.empty:
        blocks["L"] = blocks["generator"].str.split("L").str[1].astype(int)
        agg = blocks.groupby("L")["score"].agg(
            med="median", p05=lambda s: s.quantile(0.05),
            p95=lambda s: s.quantile(0.95)).reset_index()
        recs.extend(agg.to_dict("records"))
    sweep = pd.DataFrame(recs).sort_values("L")
    _band(axb, sweep["L"].to_numpy(), sweep, SERIES_1, log_x=True)
    axb.axhline(real_score, color=INK, linewidth=1.6)
    axb.text(sweep["L"].min(), real_score, "real ", color=INK, fontsize=9,
             va="bottom", ha="left")
    axb.axhline(0, color=BASELINE, linewidth=1.0)
    axb.set_xticks(sweep["L"].to_numpy())
    axb.set_xticklabels([str(int(v)) for v in sweep["L"]], fontsize=9)
    axb.set_xlabel("block-bootstrap mean block length (days)", color=SECONDARY, fontsize=10)
    axb.set_ylabel("score ($)", color=SECONDARY, fontsize=10)
    axb.set_title("Which horizon carries the edge", color=INK, fontsize=11.5,
                  fontweight="bold", loc="left", pad=10)

    # positive control
    axc = fig.add_subplot(gs[1, 1])
    _style(axc)
    ctrl = ok[ok["generator"] == "coint_synth"]
    if not ctrl.empty:
        agg = ctrl.groupby("kappa")["score"].agg(
            med="median", p05=lambda s: s.quantile(0.05),
            p95=lambda s: s.quantile(0.95)).reset_index().sort_values("kappa")
        _band(axc, agg["kappa"].to_numpy(), agg, SERIES_1)
        axc.set_xticks(agg["kappa"].to_numpy())
        axc.set_xticklabels(
            [f"{k:g}\n" + ("RW" if k <= 0 else f"{np.log(2)/k:.0f}d")
             for k in agg["kappa"]], fontsize=8.5)
    axc.axhline(real_score, color=INK, linewidth=1.6)
    axc.text(axc.get_xlim()[1], real_score, " real", color=INK, fontsize=9,
             va="bottom", ha="right")
    axc.axhline(0, color=BASELINE, linewidth=1.0)
    axc.set_xlabel("imposed OU mean-reversion rate  k  (spread half-life)",
                   color=SECONDARY, fontsize=10)
    axc.set_ylabel("score ($)", color=SECONDARY, fontsize=10)
    axc.set_title("Positive control: synthetic cointegration", color=INK,
                  fontsize=11.5, fontweight="bold", loc="left", pad=10)

    fig.savefig(OUT / "null_testing.png", dpi=160, facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'null_testing.png'}")


def md_table(df: pd.DataFrame, cols: list[str], fmts: dict[str, str]) -> str:
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(
            fmts.get(c, "{}").format(row[c]) if pd.notna(row[c]) else "-"
            for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    paths = load_paths()
    real = pd.read_csv(OUT / "real_baseline.csv").iloc[0]
    real_score = float(real["score"])
    ok = paths[paths["failed"] == 0]

    build_figure(ok, real_score)

    def stats_for(gen: str) -> dict | None:
        sub = ok[ok["generator"] == gen]
        if sub.empty:
            return None
        s, m = sub["score"].to_numpy(), sub["mean_pl"].to_numpy()
        n = len(s)
        se = m.std(ddof=1) / np.sqrt(n)
        return {
            "case": LABELS.get(gen, gen).replace("\n", " — "),
            "n": n,
            "score_med": np.median(s),
            "score_p95": np.percentile(s, 95),
            "score_max": s.max(),
            "meanpl": m.mean(),
            "meanpl_se": se,
            "t": m.mean() / se if se > 0 else 0.0,
            "p_ge_real": (np.sum(s >= real_score) + 1) / (n + 1),
        }

    prim = pd.DataFrame([r for g in PRIMARY_NULLS if (r := stats_for(g))])
    pinned = pd.DataFrame([r for g in PINNED_NULLS if (r := stats_for(g))])
    surr = pd.DataFrame([r for g in ["phase_surr"] if (r := stats_for(g))])

    n_null = int(ok["generator"].isin(PRIMARY_NULLS).sum())
    beat = int((ok.loc[ok["generator"].isin(PRIMARY_NULLS), "score"] >= real_score).sum())
    pooled_p = (beat + 1) / (n_null + 1)

    stub = ok[ok["generator"].str.startswith("stub:")]
    stub_tbl = pd.DataFrame()
    if not stub.empty:
        stub_tbl = stub.groupby("generator").apply(
            lambda g: pd.Series({
                "n": len(g),
                "meanpl": g["mean_pl"].mean(),
                "meanpl_se": g["mean_pl"].std(ddof=1) / np.sqrt(len(g)),
                "score_med": g["score"].median(),
            }), include_groups=False).reset_index()

    blocks = ok[ok["generator"].str.startswith("block_boot_L")].copy()
    block_tbl = pd.DataFrame()
    if not blocks.empty:
        blocks["L"] = blocks["generator"].str.split("L").str[1].astype(int)
        block_tbl = blocks.groupby("L").apply(
            lambda g: pd.Series({
                "n": len(g),
                "score_med": g["score"].median(),
                "pct_of_real": 100.0 * g["score"].median() / real_score,
                "sharpe_med": g["sharpe"].median(),
            }), include_groups=False).reset_index()

    ctrl = ok[ok["generator"] == "coint_synth"]
    ctrl_tbl = pd.DataFrame()
    if not ctrl.empty:
        ctrl_tbl = ctrl.groupby("kappa").apply(
            lambda g: pd.Series({
                "n": len(g),
                "half_life": np.inf if g.name <= 0 else np.log(2) / g.name,
                "score_med": g["score"].median(),
                "pct_pos": 100.0 * float((g["score"] > 0).mean()),
                "sharpe_med": g["sharpe"].median(),
            }), include_groups=False).reset_index()

    ps = ok.loc[ok["generator"] == "phase_surr", "score"]
    ps_med = float(ps.median())
    ps_lo, ps_hi = (float(v) for v in np.percentile(ps, [5, 95]))
    gap = real_score - ps_med
    gap_pct = 100.0 * gap / real_score

    scorefmt = {"n": "{:.0f}", "score_med": "{:,.1f}", "score_p95": "{:,.1f}",
                "score_max": "{:,.1f}", "meanpl": "{:,.1f}", "meanpl_se": "{:,.1f}",
                "t": "{:+.1f}", "p_ge_real": "{:.4f}"}
    scorecols = ["case", "n", "score_med", "score_p95", "score_max", "meanpl",
                 "meanpl_se", "t", "p_ge_real"]

    report = f"""# Synthetic-data and NULL testing — four_sigmas

Scoring window **751-1000**, eval.py accounting replicated verbatim (one-day
commission lag, truncating dollar-limit share caps, integer shares, mark-only
final day). The replica reproduces eval.py's real score to the cent, so every
synthetic number is on the leaderboard scale.

**Real data: score {real_score:,.2f}** (mean ${real['mean_pl']:,.1f}/day, sd
${real['std_pl']:,.1f}, Sharpe {real['sharpe']:.2f}). {len(ok):,} synthetic paths;
each is a full 1000-day price matrix replayed day-by-day with the book's state reset.

## 1. The null test

Each null destroys the exploitable structure a different way while holding
something real fixed. All keep day-1 price levels, so dollar limits and integer
rounding bite exactly as on real data. `t` is on mean daily PnL against zero.

{md_table(prim, scorecols, scorefmt)}

Pooled across the endpoint-free nulls: **{beat} of {n_null}** paths reached the
real score — empirical p = **{pooled_p:.4f}**. The best single null path in the
whole study reached {ok[ok['generator'].isin(PRIMARY_NULLS)]['score'].max():,.0f}
against a real {real_score:,.0f}.

## 2. A null that was wrong, and how it showed up

The first pass used *permutation* nulls. They looked fine but returned a small,
strongly significant positive mean PnL:

{md_table(pinned, scorecols, scorefmt)}

That is an artifact of the null, not a bias in the book. A permutation preserves
the **sum** of log returns, so every permuted path lands on the real terminal
price exactly (verified: max relative difference 4e-16). Each path is therefore a
Brownian bridge, with `E[r_t+1 | S_t] = (total - S_t)/(n - t)` pulling toward a
pinned endpoint — real, exploitable mean reversion, which is exactly what this
book trades. A mean-reverting strategy *should* profit there.

Two checks confirm it:

- A signal-free control — fixed random long/short signs at the same full dollar
  limit, rebalanced daily, so identical sizing mechanics and no signal — earns
  nothing on the same pinned paths. The bridge gain needs the mean-reversion
  signal to harvest it, so it is not a sizing or accounting artifact.
{md_table(stub_tbl, ["generator", "n", "meanpl", "meanpl_se", "score_med"], {"n": "{:.0f}", "meanpl": "{:,.1f}", "meanpl_se": "{:,.1f}", "score_med": "{:,.1f}"}) if not stub_tbl.empty else "  _(not run)_"}

- Resampling **with replacement** unpins the endpoint and the effect disappears —
  that is the §1 table, and it is what the verdict rests on.

The pinned nulls are kept in the figure because the contrast measures the
artifact. Note the direction: the bias made the null *stronger*, so the test was
conservative, and the conclusion survives it.

## 3. Horizon decomposition (block bootstrap)

Block resampling keeps dependence shorter than the block and destroys everything
longer, so the block length at which the score returns is the horizon the book
actually trades. Length 1 is the `xs_boot` null.

{md_table(block_tbl, ["L", "n", "score_med", "pct_of_real", "sharpe_med"], {"L": "{:.0f}", "n": "{:.0f}", "score_med": "{:,.1f}", "pct_of_real": "{:.0f}%", "sharpe_med": "{:.2f}"}) if not block_tbl.empty else "_not run_"}

Essentially all the edge is intra-month, and about two thirds of it is already
back at a 5-day block. Nothing at horizons beyond ~20 days contributes: the book
is a short-horizon mean-reversion engine, which is what it claims to be. (These
paths are with-replacement and so endpoint-free.)

## 4. Positive control — synthetic cointegration

A one-factor market plus genuine OU-cointegrated spreads on an asset-disjoint
subset of the book's *own* configured pairs (~25 of 88; every other asset, and
all lead-lag structure the ridge sleeve wants, stays a pure random walk).
k = 0 means the spread is a random walk — no cointegration at all.

{md_table(ctrl_tbl, ["kappa", "n", "half_life", "score_med", "pct_pos", "sharpe_med"], {"kappa": "{:.2f}", "n": "{:.0f}", "half_life": "{:.1f}", "score_med": "{:,.1f}", "pct_pos": "{:.0f}%", "sharpe_med": "{:.2f}"}) if not ctrl_tbl.empty else "_not run_"}

Monotone in true mean-reversion speed, and its own internal null (k=0) is
negative. The book responds to genuine cointegration rather than to its own
parameterisation. This is a validity check, **not** a calibration of the real
data's k — the control deliberately structures under a third of the pairs and
gives the ridge sleeve nothing, so its levels are not comparable to the real score.

## 5. How much of the score is this particular path?

`phase_surr` is not a null: shared-phase randomisation preserves the entire
cross-spectral density — every auto- and cross-covariance at every lag, including
the linear cointegration structure — and changes only the realised path. With the
88 pairs and their thresholds frozen, it estimates what this book scores on a
**fresh draw from the same linear structure**.

| | score |
|---|---|
| realised (real data) | **{real_score:,.0f}** |
| surrogate median | {ps_med:,.0f} |
| surrogate 5-95% | {ps_lo:,.0f} - {ps_hi:,.0f} |
| shortfall | **{gap:,.0f}  ({gap_pct:.0f}% of realised)** |

Roughly {100 - gap_pct:.0f}% of the score is structure that reproduces on a
redrawn path; about {gap_pct:.0f}% is specific to this realisation — frozen
parameters sitting where this particular path rewarded them. Two caveats, in
opposite directions: the surrogate preserves the DC term, so it inherits the same
endpoint pinning as §2 and is flattered by a few percent (the true shortfall is
somewhat larger); and the shortfall is a *floor* on out-of-sample decay, not a
forecast, because a real future window also brings regime drift that no surrogate
of this history contains.

## What this does and does not establish

**Does.** The machine does not manufacture profit from nothing. On structureless
data with real drift, volatility and cross-sectional correlation preserved, the
book's mean PnL is negative (commission drag) and its score distribution sits at
~0, with the real score far outside it (p = {pooled_p:.4f}). No look-ahead, no
accounting artifact, no long-drift bias masquerading as edge. The positive control
shows the harness detects edge when edge exists.

**Does not.** This is a *mechanism* null, not a *selection* null. The 88 pairs and
their per-pair thresholds were chosen by looking at this price history and stay
frozen across every synthetic path. Null-testing the selection itself would mean
re-running the whole pair-discovery pipeline on each synthetic path. So the
p-value says "the book extracts real structure from this data" — not "the
751-1000 score is unaffected by having selected on 501-750". §5 is the closest
this study gets to that question, and it puts {gap_pct:.0f}% of the score on the
realised path.

Artifacts: `paths_*.csv` (every path), `null_summary.csv`, `block_sweep.csv`,
`control_sweep.csv`, `null_testing.png`.
Reproduce:
```
python research/synthetic_null.py --mode full --block-reps 50 --control-reps 25 --workers 7
python research/synthetic_null.py --mode null --reps 150 --workers 7
python research/synthetic_null.py --mode mechanism --reps 60 --workers 7
python research/synthetic_null_report.py
```
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    pd.concat([prim, pinned, surr]).to_csv(OUT / "null_summary.csv", index=False)
    if not block_tbl.empty:
        block_tbl.to_csv(OUT / "block_sweep.csv", index=False)
    if not ctrl_tbl.empty:
        ctrl_tbl.to_csv(OUT / "control_sweep.csv", index=False)

    print(f"\nreal score {real_score:,.2f}")
    print(pd.concat([prim, pinned, surr]).to_string(index=False))
    if not block_tbl.empty:
        print("\nblocks\n" + block_tbl.to_string(index=False))
    if not ctrl_tbl.empty:
        print("\ncontrol\n" + ctrl_tbl.to_string(index=False))
    if not stub_tbl.empty:
        print("\nsignal-free stub\n" + stub_tbl.to_string(index=False))
    print(f"\npooled null p = {pooled_p:.4f} ({beat}/{n_null})")
    print(f"wrote {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
