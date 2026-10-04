# Whole-vector EV conflict model

> **Source:** `analysis_outputs/whole_vector_ev/README.md` in the team's working repository (relocated here for the public release).

The model was developed under a locked three-way split:

| Data | Role |
|---|---|
| Days 1–500 | Fit expected gross PnL, identity shrinkage and variance |
| Days 501–750 | Select prior strength and minimum net-edge threshold |
| Days 751–1000 | One held-out evaluation |

Each pair contributes its complete beta-scaled two-leg position vector. Ridge
contributes one single-asset vector per active forecast. For every proposal:

1. Expected gross next-day PnL is estimated using a monotone family slope and
   a strongly shrunk pair/asset intercept.
2. Known entry/rebalance cost and a one-way liquidation reserve are deducted.
3. The proposal is retained only when net EV exceeds the selected uncertainty
   buffer.
4. Retained vectors receive `net EV / forecast variance` weights.
5. Complete weighted vectors are summed; final asset directions remain
   bang-bang and instrument limits are applied last.

The validation-selected configuration was:

- Prior observations: **200**
- Minimum edge: **0.01 × forecast standard deviation**

## Results

| Period | Existing method | Whole-vector EV | Change |
|---|---:|---:|---:|
| Validation 501–625 | 853.57 | 1062.17 | +208.60 |
| Validation 626–750 | 1049.43 | 1059.51 | +10.09 |
| Validation 501–750 | 951.52 | 1060.93 | +109.42 |
| Held out 751–1000 | 886.39 | 845.95 | **−40.45** |

The held-out result rejects this model as the production replacement.
Although volatility fell enough to raise held-out Sharpe from 8.11 to 9.67,
mean daily PnL fell from \$899.86 to \$855.00. The competition score primarily
rewards mean PnL once Sharpe is already high, so the risk reduction did not
compensate for lost deployed edge.

`four_sigmas.py` therefore remains unchanged. `four_sigmas_ev.py` is the
frozen experimental implementation and must not be promoted based on further
inspection of days 751–1000.

## Post-test diagnosis

The model changes approximately 10–12% of asset decisions, primarily by
flipping the baseline direction; it drops only about 1% and never adds an
asset that the baseline did not already trade.

| Window | Score change | Gross PnL from flips | Pair actual/predicted gross PnL | Ridge actual/predicted gross PnL |
|---|---:|---:|---:|---:|
| Validation 501–625 | +208.60 | +\$28,655 | \$50.17 / \$45.32 | \$16.31 / \$13.51 |
| Validation 626–750 | +10.09 | −\$426 | \$41.25 / \$45.99 | \$27.74 / \$13.88 |
| OOS 751–875 | −157.07 | −\$19,142 | \$32.48 / \$46.05 | \$37.60 / \$13.69 |
| OOS 876–1000 | +78.05 | +\$7,508 | \$29.53 / \$45.34 | \$8.81 / \$13.58 |

The calibration learned that an active pair proposal was worth about \$45
gross while an active ridge proposal was worth about \$14. It therefore
weighted the complete pair vectors much more heavily. During OOS1 the
opposite regime arrived: realised pair edge fell to \$32 while ridge edge rose
to \$38. The frozen model continued trusting its training relationship and
flipped profitable baseline/ridge decisions in the wrong direction.

The apparent validation strength was also uneven. Almost the entire
validation improvement occurred during 501–625; by 626–750 the benefit had
already fallen from +208.60 to +10.09.

This is not a parameter-free method. The frozen runtime contains:

- 88 pair-specific expected-PnL offsets and 88 variances;
- 51 ridge-asset offsets and 51 variances;
- two global means, slopes and clipping ranges;
- a selected 200-observation shrinkage prior;
- a selected 0.01-standard-deviation edge threshold.

That is roughly 290 learned or selected scalar quantities. Furthermore,
the correlation between predicted and realised proposal PnL was only
0.02–0.08 for pairs and −0.02–0.11 for ridge across the fixed windows.
Consequently the estimated EV was mostly a historical average, not a strong
conditional day-by-day forecast.

Detailed evidence is in `oos_portfolio_diagnosis.csv` and
`oos_calibration_diagnosis.csv`.
