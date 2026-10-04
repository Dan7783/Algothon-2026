# Pair-filter techniques applied to EV combination

> **Source:** `analysis_outputs/adaptive_ev_pair_filter/README.md` in the team's working repository (relocated here for the public release).

Three causal adaptations were tested. Configuration selection used only
days 1–750; held-out results could not select or modify a setting.

## 1. Two adaptive family states

One state tracked pair proposal edge and one tracked ridge proposal edge.
Each used:

- EWMA mean edge and forecast variance;
- strength-relative proposal scaling;
- two-sided CUSUM forecast-error monitoring;
- uncertainty inflation and mean reset on an alarm.

The best setting (`half_life=20`, `CUSUM h=4`) improved three of four
development folds but failed validation 626–750 by **−201.58**. It was
rejected before OOS and days 751–1000 were not loaded for it.

## 2. Independent pair and ridge-asset states

Every pair and ridge asset received an adaptive mean/variance state while all
states shared the same EWMA and CUSUM settings. Sparse estimates were shrunk
toward the family state.

The best setting (`half_life=40`, `CUSUM h=8`) produced:

| Fold | Change versus baseline |
|---|---:|
| Train 251–375 | +70.40 |
| Train 376–500 | −41.05 |
| Validation 501–625 | +200.46 |
| Validation 626–750 | −25.52 |

It failed the four-fold promotion rule and OOS remained closed.

## 3. Static EV with a causal baseline fallback

Both the static EV and baseline books were marked as shadow portfolios. An
EWMA tracked strictly-prior realised EV-minus-baseline PnL:

- positive state: deploy EV;
- negative state: deploy baseline;
- both shadow books continue running so the gate can switch back.

An 80-day half-life passed both validation halves:

| Validation fold | Change |
|---|---:|
| 501–625 | +187.65 |
| 626–750 | +10.09 |
| Full 501–750 | +98.93 |

The setting was frozen before retrospective OOS confirmation:

| OOS fold | Baseline | Gated EV | Change |
|---|---:|---:|---:|
| 751–875 | 1190.27 | 1107.57 | −82.69 |
| 876–1000 | 580.61 | 540.57 | −40.03 |
| Full 751–1000 | 886.39 | 824.70 | **−61.69** |

The gate used EV only 39.0% of OOS. It reacted too slowly to the initial EV
failure, then remained defensive when EV recovered during the second half.

## Conclusion

The pair-filter techniques are reasonable but do not repair the EV model
under forward evidence. The latent "which combiner is better" regime changes
faster than a validation-stable smoother can identify from noisy daily PnL.
Faster smoothers switch often and degraded validation; slower smoothers pass
validation but lag OOS.

No adaptive EV variant is promoted. `four_sigmas.py` remains unchanged.
