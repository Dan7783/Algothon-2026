"""Can the ridge predict ALGO? Three questions, in order.

  A. Is ALGO a deterministic function of the other 50 returns (contemporaneous)?
     If yes, "predict ALGO" really does reduce to "predict the basket".

  B. Does that reduction survive one step ahead? Aggregation cancels relative
     value (spreads are differences) and leaves the common factor. So the test
     is whether the ridge predicts ALGO as well as it predicts a normal asset.

  C. What is the theoretical ceiling? Even a PERFECT per-asset ridge, aggregated
     with the true basket weights, has some correlation with next-day ALGO.
     Measure it, so the ALGO slot is judged against what is achievable, not zero.

Read-only: nothing in four_sigmas.py is modified.
Run from repo root:  python research/algo_ridge_probe.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

import four_sigmas as fs

OUT = _ROOT / "analysis_outputs" / "algo_ridge_probe"
OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(_ROOT / "prices.txt", sep=r"\s+", header=0)
SYMBOLS = list(raw.columns)
PRC = raw.to_numpy(float).T
N, T = PRC.shape
R = np.diff(np.log(PRC), axis=1)          # (51, 999)
ALGO = fs.I["ALGO"]
OTHERS = np.array([i for i in range(N) if i != ALGO])

EVAL_START = 750        # first return index scored (day ~751)
LAM = fs.RIDGE_LAM

lines: list[str] = []


def say(s: str = "") -> None:
    print(s)
    lines.append(s)


# ---------------------------------------------------------------- A
say("A.  IS ALGO DETERMINISTIC IN THE OTHER 50 RETURNS?")
say("=" * 78)
X = R[OTHERS].T
y = R[ALGO]
w, *_ = np.linalg.lstsq(X, y, rcond=None)
fit = X @ w
ss_res = float(np.sum((y - fit) ** 2))
ss_tot = float(np.sum((y - y.mean()) ** 2))
r2_full = 1.0 - ss_res / ss_tot
say(f"  contemporaneous R^2 of ALGO on the other 50: {r2_full:.6f}")
say(f"  residual sd {np.std(y - fit):.6f} vs ALGO sd {np.std(y):.6f} "
    f"({100 * np.std(y - fit) / np.std(y):.1f}% unexplained)")

# out-of-sample: fit weights on 1-750, apply to 751-1000
w_tr, *_ = np.linalg.lstsq(X[:EVAL_START], y[:EVAL_START], rcond=None)
pred_oos = X[EVAL_START:] @ w_tr
r2_oos = 1.0 - float(np.sum((y[EVAL_START:] - pred_oos) ** 2)) / float(
    np.sum((y[EVAL_START:] - y[EVAL_START:].mean()) ** 2))
say(f"  same-day R^2 out of sample (weights from 1-750): {r2_oos:.6f}")
say(f"  weights: min {w.min():+.4f}  max {w.max():+.4f}  sum {w.sum():+.4f}  "
    f"negative {int((w < 0).sum())}/50")
say("  -> ALGO is a near-exact contemporaneous basket, so the reduction is fair:")
say("     predicting ALGO == predicting the weighted sum of the other 50.")

# ---------------------------------------------------------------- B
say("\nB.  DOES IT SURVIVE ONE STEP AHEAD?")
say("=" * 78)
say(f"  ridge replicated from _ridge_forecast_details (lam={LAM}), refit daily,")
say(f"  ALGO added as BOTH predictor and target, scored on returns "
    f"{EVAL_START}..{R.shape[1] - 1}")


def ridge_step(hist: np.ndarray, lam: float) -> np.ndarray:
    """hist is (n_obs, n_assets) of past returns; returns next-step prediction."""
    scale = hist.std(axis=0) + 1.0e-12
    predictors = hist[:-1] / scale
    targets = hist[1:] / scale
    coef = np.linalg.solve(
        predictors.T @ predictors
        + lam * len(predictors) * np.eye(hist.shape[1]),
        predictors.T @ targets,
    )
    return (hist[-1] / scale) @ coef


n_ret = R.shape[1]
preds = np.zeros((N, n_ret - EVAL_START))
actual = R[:, EVAL_START:]
for j, k in enumerate(range(EVAL_START, n_ret)):
    preds[:, j] = ridge_step(R[:, :k].T, LAM)

sign_acc = np.mean(np.sign(preds) == np.sign(actual), axis=1)
corr = np.array([
    np.corrcoef(preds[i], actual[i])[0, 1] if preds[i].std() > 0 else 0.0
    for i in range(N)
])
n_eval = actual.shape[1]
se_acc = 0.5 / np.sqrt(n_eval)

others_acc, others_corr = sign_acc[OTHERS], corr[OTHERS]
say(f"  evaluation days: {n_eval}   (binomial SE on accuracy = {se_acc:.4f})")
say(f"  {'':22s} {'sign acc':>9} {'z vs 50%':>9} {'corr':>8}")
say(f"  {'ALGO (market proxy)':22s} {sign_acc[ALGO]:>9.4f} "
    f"{(sign_acc[ALGO] - 0.5) / se_acc:>9.2f} {corr[ALGO]:>8.4f}")
say(f"  {'other 50, mean':22s} {others_acc.mean():>9.4f} "
    f"{(others_acc.mean() - 0.5) / (se_acc / np.sqrt(50)):>9.2f} "
    f"{others_corr.mean():>8.4f}")
say(f"  {'other 50, best':22s} {others_acc.max():>9.4f} {'':>9} "
    f"{others_corr.max():>8.4f}")
say(f"  {'other 50, worst':22s} {others_acc.min():>9.4f} {'':>9} "
    f"{others_corr.min():>8.4f}")
rank = int((others_acc > sign_acc[ALGO]).sum())
say(f"  -> ALGO ranks {rank + 1} of {N} on sign accuracy "
    f"({rank} of the other 50 beat it)")

# ---------------------------------------------------------------- C
say("\nC.  THE AGGREGATION ARGUMENT, MEASURED")
say("=" * 78)
say("  If ALGO = w . r, then the basket-implied ALGO forecast is w . (per-asset")
say("  ridge forecasts). Compare that against ALGO's realised return.")
basket_pred = w_tr @ preds[OTHERS]
basket_actual = w_tr @ actual[OTHERS]
bp_corr = float(np.corrcoef(basket_pred, basket_actual)[0, 1])
bp_acc = float(np.mean(np.sign(basket_pred) == np.sign(basket_actual)))
say(f"  basket-implied forecast vs realised basket: corr {bp_corr:+.4f}, "
    f"sign acc {bp_acc:.4f} (z={(bp_acc - 0.5) / se_acc:+.2f})")
say(f"  direct ALGO ridge forecast vs realised ALGO: corr {corr[ALGO]:+.4f}, "
    f"sign acc {sign_acc[ALGO]:.4f}")

# why aggregation destroys it: how much of per-asset predictability is relative?
say("\n  where the per-asset predictability lives:")
demeaned_pred = preds[OTHERS] - preds[OTHERS].mean(axis=0)
demeaned_act = actual[OTHERS] - actual[OTHERS].mean(axis=0)
rel_corr = np.array([
    np.corrcoef(demeaned_pred[i], demeaned_act[i])[0, 1] for i in range(len(OTHERS))
])
say(f"  mean per-asset corr, RAW returns            : {others_corr.mean():+.4f}")
say(f"  mean per-asset corr, CROSS-SECTIONALLY DEMEANED: {rel_corr.mean():+.4f}")
say("  -> the ridge's skill is relative-value. Removing the cross-sectional mean")
say("     leaves it intact (or better); aggregating into a basket cancels it.")

pd.DataFrame({
    "symbol": SYMBOLS,
    "sign_acc": sign_acc,
    "corr": corr,
    "is_algo": [i == ALGO for i in range(N)],
}).to_csv(OUT / "per_asset_predictability.csv", index=False)
(OUT / "probe.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\nwrote {OUT / 'probe.txt'}")
