"""Sequential tuning of the sticky through-origin Kalman pair filter.

Tunes ``pair_seg_filter.PairSegFilter`` on the log prices of the fixed atlas
pairs, training window only (first ``TRAIN_DAYS`` days).  Calibration follows
the same policy as the other filters in this repository:

1. Kalman speed first, segmentation structurally off.  Per pair, a grid over
   ``(q_beta, ewma_lambda)`` is scored with a causal prequential Gaussian loss.
   The one-standard-error rule keeps the stickiest ``q_beta`` whose loss is
   statistically indistinguishable from the minimum.
2. Speeds frozen, segmentation knobs second.  One global grid over
   ``(cusum_k, h, target_seg, min_seg_len, snap_overshoot)`` is scored by the
   same loss summed across pairs; within one standard error the configuration
   with the fewest cuts wins, so segmentation must earn every snap.

THE LOSS IS MULTI-HORIZON ON FROZEN BETAS.  A through-origin levels filter can
make its one-step residual arbitrarily small by letting beta absorb the
stationary spread; scored that way every pair selects the fastest grid speed
(observed empirically: all pairs pinned at the q upper bound, zero cuts).  The
hedge the strategy holds is the beta published at ``t - h``, so the loss is
the Gaussian score of ``(y_t - y_{t-h}) - beta_{t-h} (x_t - x_{t-h})`` for
``h`` in ``HORIZONS``, each horizon standardized by its own prequential EWMA
variance.  Chasing wins one-step but mis-hedges multi-day holds; a frozen
wrong beta loses at every horizon.  Standard errors use 20-day block means
because overlapping h-step errors are serially dependent.

Never selects on trading PnL.  Outputs land in ``cusum_output/pair_seg_tuning``.
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

import json
import time
from math import log, sqrt
from pathlib import Path

import numpy as np

from adaptive_beta_pairs import PAIR_I, PAIR_J
from calibrate import load_prices
from research.pairs_analysis.pair_seg_filter import (
    PairSegConfig,
    PairSegFilter,
    plot_smoothed_vs_raw,
    plot_snap_speed_comparison,
    run_pair_seg,
)

TRAIN_DAYS = 500
BURN_IN = 60  # innovations scored only after init/EWMA transients
HORIZONS = (1, 10, 20)  # hedge holding horizons scored with frozen beta
SE_BLOCK = 20  # block length for serially-dependent loss standard errors
NO_CUT_WARMUP = 10**9  # CUSUM block never runs -> pure sticky Kalman

Q_GRID = [0.0] + list(np.geomspace(1e-10, 1e-3, 15))
LAMBDA_GRID = [0.88, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99]

K_GRID = [0.25, 0.5, 1.0]
H_GRID = [6.0, 10.0, 15.0]
TARGET_SEG_GRID = [30, 40, 60]
MIN_SEG_GRID = [20, 40, 60]
OVERSHOOT_GRID = [1.0, 1.25]

OUT_DIR = Path("cusum_output/pair_seg_tuning")


def prequential_run(
    y: np.ndarray,
    x: np.ndarray,
    cfg: PairSegConfig,
) -> tuple[np.ndarray, int]:
    """Per-step multi-horizon frozen-beta loss and the number of cuts.

    The error at ``(t, h)`` uses only the beta published at ``t - h`` and a
    variance estimated from errors before ``t``; nothing from ``(t - h, t]``
    leaks into its own forecast.
    """
    filt = PairSegFilter(cfg)
    lam = cfg.ewma_lambda
    T = len(y)
    beta_path = np.empty(T)
    variance: dict[int, float | None] = {h: None for h in HORIZONS}
    per_step: list[float] = []
    cuts = 0
    for t in range(T):
        out = filt.step(float(y[t]), float(x[t]))
        if out.cut:
            cuts += 1
        beta_path[t] = out.beta
        step_losses = []
        for h in HORIZONS:
            if t < h:
                continue
            error = (y[t] - y[t - h]) - beta_path[t - h] * (x[t] - x[t - h])
            prior = variance[h]
            if t >= BURN_IN and prior is not None and prior > 1e-18:
                step_losses.append(
                    0.5 * (log(prior) + error * error / prior)
                )
            variance[h] = (
                error * error
                if prior is None
                else lam * prior + (1.0 - lam) * error * error
            )
        if t >= BURN_IN and len(step_losses) == len(HORIZONS):
            per_step.append(sum(step_losses) / len(HORIZONS))
    return np.asarray(per_step), cuts


def _block_means(losses: np.ndarray, block: int = SE_BLOCK) -> np.ndarray:
    blocks = len(losses) // block
    if blocks < 2:
        return losses
    return losses[: blocks * block].reshape(blocks, block).mean(axis=1)


def _block_se(losses: np.ndarray) -> float:
    means = _block_means(losses)
    return float(np.std(means, ddof=1)) / sqrt(len(means))


def tune_speed(y: np.ndarray, x: np.ndarray) -> dict:
    """Stage 1: per-pair (q_beta, ewma_lambda) with segmentation off."""
    rows = []
    for q in Q_GRID:
        for lam in LAMBDA_GRID:
            cfg = PairSegConfig(q_beta=q, ewma_lambda=lam, warmup=NO_CUT_WARMUP)
            losses, _ = prequential_run(y, x, cfg)
            rows.append((q, lam, float(losses.mean()), losses))
    best = min(rows, key=lambda r: r[2])
    se = _block_se(best[3])
    eligible = [r for r in rows if r[2] <= best[2] + se]
    stickiest_q = min(r[0] for r in eligible)
    at_q = [r for r in eligible if r[0] == stickiest_q]
    # Same one-SE conservatism on lambda: the longest variance memory that is
    # indistinguishable from the best row at the sticky speed.
    best_lam = min(at_q, key=lambda r: r[2])
    lam_se = _block_se(best_lam[3])
    chosen = max(
        (r for r in at_q if r[2] <= best_lam[2] + lam_se),
        key=lambda r: r[1],
    )
    default_losses, _ = prequential_run(
        y, x, PairSegConfig(warmup=NO_CUT_WARMUP)
    )
    return {
        "q_beta": float(chosen[0]),
        "ewma_lambda": float(chosen[1]),
        "loss": float(chosen[2]),
        "min_loss": float(best[2]),
        "min_q_beta": float(best[0]),
        "standard_error": se,
        "default_loss": float(default_losses.mean()),
        "no_cut_losses": chosen[3],
    }


def segment_config(speed: dict, k: float, h: float, ts: int, msl: int, os_: float) -> PairSegConfig:
    return PairSegConfig(
        q_beta=speed["q_beta"],
        ewma_lambda=speed["ewma_lambda"],
        cusum_k=k,
        h=h,
        target_seg=ts,
        min_seg_len=msl,
        snap_overshoot=os_,
    )


def tune_segmentation(
    pair_series: list[tuple[np.ndarray, np.ndarray]],
    speeds: list[dict],
) -> dict:
    """Stage 2: one global segmentation policy over all pairs."""
    combos = []
    for k in K_GRID:
        for h in H_GRID:
            for ts in TARGET_SEG_GRID:
                for msl in MIN_SEG_GRID:
                    for os_ in OVERSHOOT_GRID:
                        combos.append((k, h, ts, msl, os_))
    results = []
    for combo in combos:
        k, h, ts, msl, os_ = combo
        all_losses = []
        block_means = []
        pair_cuts = []
        for (y, x), speed in zip(pair_series, speeds):
            losses, cuts = prequential_run(
                y, x, segment_config(speed, k, h, ts, msl, os_)
            )
            all_losses.append(losses)
            block_means.append(_block_means(losses))
            pair_cuts.append(cuts)
        concat = np.concatenate(all_losses)
        pooled_blocks = np.concatenate(block_means)
        results.append(
            {
                "cusum_k": k,
                "h": h,
                "target_seg": ts,
                "min_seg_len": msl,
                "snap_overshoot": os_,
                "loss": float(concat.mean()),
                "cuts": int(sum(pair_cuts)),
                "pair_cuts": pair_cuts,
                "block_se": float(np.std(pooled_blocks, ddof=1))
                / sqrt(len(pooled_blocks)),
            }
        )
    best = min(results, key=lambda r: r["loss"])
    se = best["block_se"]
    eligible = [r for r in results if r["loss"] <= best["loss"] + se]
    chosen = min(
        eligible,
        key=lambda r: (
            r["cuts"],
            -r["min_seg_len"],
            -r["cusum_k"],
            -r["h"],
            r["snap_overshoot"],
            r["loss"],
        ),
    )
    return {
        "chosen": chosen,
        "best": best,
        "standard_error": se,
        "n_eligible": len(eligible),
        "n_combos": len(results),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_prices()
    tickers = list(panel.columns)
    log_prices = np.log(panel.to_numpy(dtype=float).T)[:, :TRAIN_DAYS]

    pairs = list(zip(PAIR_I.tolist(), PAIR_J.tolist()))
    pair_series = [(log_prices[i], log_prices[j]) for i, j in pairs]
    pair_names = [f"{tickers[i]}/{tickers[j]}" for i, j in pairs]

    print(f"Tuning on log prices, days 1..{TRAIN_DAYS}, {len(pairs)} pairs")
    print(f"Loss horizons {HORIZONS}, burn-in {BURN_IN}, SE block {SE_BLOCK}")
    t0 = time.perf_counter()
    speeds = []
    for name, (y, x) in zip(pair_names, pair_series):
        speed = tune_speed(y, x)
        speeds.append(speed)
        print(
            f"  stage1 {name:>10}: q_beta={speed['q_beta']:.3e} "
            f"lambda={speed['ewma_lambda']:.2f} loss={speed['loss']:+.4f} "
            f"(min {speed['min_loss']:+.4f} at q={speed['min_q_beta']:.3e}, "
            f"SE {speed['standard_error']:.4f}, default {speed['default_loss']:+.4f})"
        )
    print(f"stage 1 done in {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    seg = tune_segmentation(pair_series, speeds)
    chosen = seg["chosen"]
    no_cut_total = float(
        np.concatenate([s["no_cut_losses"] for s in speeds]).mean()
    )
    print(f"stage 2 done in {time.perf_counter() - t1:.1f}s")
    print(
        f"  segmentation: cusum_k={chosen['cusum_k']} h={chosen['h']} "
        f"target_seg={chosen['target_seg']} min_seg_len={chosen['min_seg_len']} "
        f"snap_overshoot={chosen['snap_overshoot']}"
    )
    print(
        f"  loss with cuts {chosen['loss']:+.4f} vs no-cut {no_cut_total:+.4f} "
        f"(best {seg['best']['loss']:+.4f}, SE {seg['standard_error']:.4f}, "
        f"{seg['n_eligible']}/{seg['n_combos']} within one SE), "
        f"total cuts {chosen['cuts']}"
    )

    tuned_cfgs = [
        segment_config(
            speed,
            chosen["cusum_k"],
            chosen["h"],
            chosen["target_seg"],
            chosen["min_seg_len"],
            chosen["snap_overshoot"],
        )
        for speed in speeds
    ]

    # ---- plots: untuned defaults vs tuned, for representative pairs ----
    tuned_runs = [
        run_pair_seg(y, x, cfg) for (y, x), cfg in zip(pair_series, tuned_cfgs)
    ]
    tuned_losses = [
        float(prequential_run(y, x, cfg)[0].mean())
        for (y, x), cfg in zip(pair_series, tuned_cfgs)
    ]
    most_cut = int(np.argmax([int(r.cut.sum()) for r in tuned_runs]))
    plot_idx = sorted({0, 2, most_cut})
    for idx in plot_idx:
        y, x = pair_series[idx]
        name = pair_names[idx].replace("/", "_").lower()
        default_run = run_pair_seg(y, x, PairSegConfig())
        plot_smoothed_vs_raw(
            y,
            x,
            tuned_runs[idx],
            y_ticker=tickers[pairs[idx][0]],
            x_ticker=tickers[pairs[idx][1]],
            path=OUT_DIR / f"{name}_tuned_beta.png",
            break_window=(280, 460),
        )
        plot_snap_speed_comparison(
            y,
            x,
            default_run,
            tuned_runs[idx],
            y_ticker=tickers[pairs[idx][0]],
            x_ticker=tickers[pairs[idx][1]],
            path=OUT_DIR / f"{name}_default_vs_tuned.png",
            break_window=(280, 460),
        )

    # ---- machine-readable tuned configuration ----
    payload = {
        "train_days": TRAIN_DAYS,
        "burn_in": BURN_IN,
        "horizons": list(HORIZONS),
        "input": "log prices",
        "segmentation": {
            "cusum_k": chosen["cusum_k"],
            "h": chosen["h"],
            "target_seg": chosen["target_seg"],
            "min_seg_len": chosen["min_seg_len"],
            "snap_overshoot": chosen["snap_overshoot"],
        },
        "pairs": [
            {
                "i": int(i),
                "j": int(j),
                "name": name,
                "q_beta": speed["q_beta"],
                "ewma_lambda": speed["ewma_lambda"],
                "loss_no_cut": speed["loss"],
                "loss_with_cuts": loss_cut,
                "loss_default": speed["default_loss"],
                "cuts": int(run.cut.sum()),
                "cut_days": np.flatnonzero(run.cut).tolist(),
            }
            for (i, j), name, speed, run, loss_cut in zip(
                pairs, pair_names, speeds, tuned_runs, tuned_losses
            )
        ],
        "comparison": {
            "loss_tuned_with_cuts": chosen["loss"],
            "loss_tuned_no_cuts": no_cut_total,
            "loss_best_combo": seg["best"]["loss"],
            "standard_error": seg["standard_error"],
            "eligible_within_one_se": seg["n_eligible"],
            "combos": seg["n_combos"],
        },
    }
    (OUT_DIR / "tuned_pair_seg.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # ---- report ----
    lines = [
        "# PairSegFilter Kalman tuning (atlas pairs, train days 1-500)",
        "",
        "Input: log prices.  Loss: causal prequential Gaussian score of the "
        f"frozen-beta hedge error at horizons {list(HORIZONS)} days "
        f"(burn-in {BURN_IN}; block-{SE_BLOCK} standard errors).  One-step "
        "levels loss is degenerate here — beta can absorb the stationary "
        "spread, and every pair pins the fastest grid speed — so the beta "
        "published at t-h must hedge the move from t-h to t.  The one-SE rule "
        "prefers sticky q in stage 1 and few cuts in stage 2.",
        "",
        "## Stage 1 — Kalman speed per pair (segmentation off)",
        "",
        "| pair | q_beta | lambda | loss | min loss | SE | default loss |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, speed in zip(pair_names, speeds):
        lines.append(
            f"| {name} | {speed['q_beta']:.3e} | {speed['ewma_lambda']:.2f} "
            f"| {speed['loss']:+.4f} | {speed['min_loss']:+.4f} "
            f"| {speed['standard_error']:.4f} | {speed['default_loss']:+.4f} |"
        )
    lines += [
        "",
        "## Stage 2 — global segmentation policy (speeds frozen)",
        "",
        f"- cusum_k `{chosen['cusum_k']}`, h `{chosen['h']}`, "
        f"target_seg `{chosen['target_seg']}`, "
        f"min_seg_len `{chosen['min_seg_len']}`, "
        f"snap_overshoot `{chosen['snap_overshoot']}`",
        f"- loss with cuts `{chosen['loss']:+.4f}` vs no-cut "
        f"`{no_cut_total:+.4f}` (grid best `{seg['best']['loss']:+.4f}`, "
        f"SE `{seg['standard_error']:.4f}`, {seg['n_eligible']} of "
        f"{seg['n_combos']} combos within one SE)",
        f"- total cuts `{chosen['cuts']}`",
        "",
        "| pair | cuts | loss with cuts | loss no cuts | cut days |",
        "|---|---|---|---|---|",
    ]
    for name, speed, run, loss_cut in zip(
        pair_names, speeds, tuned_runs, tuned_losses
    ):
        lines.append(
            f"| {name} | {int(run.cut.sum())} | {loss_cut:+.4f} "
            f"| {speed['loss']:+.4f} "
            f"| {np.flatnonzero(run.cut).tolist()} |"
        )
    lines += [
        "",
        "Artifacts: `tuned_pair_seg.json`, `*_tuned_beta.png`, "
        "`*_default_vs_tuned.png`.",
        "",
    ]
    (OUT_DIR / "tuning_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'tuned_pair_seg.json'} and tuning_report.md")


if __name__ == "__main__":
    main()
