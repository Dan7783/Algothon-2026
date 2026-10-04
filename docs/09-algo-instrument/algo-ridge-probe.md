# Is ALGO predictable through the ridge?

> **Source:** `analysis_outputs/algo_ridge_probe/probe.txt` in the team's working repository (relocated here for the public release).

```text
A.  IS ALGO DETERMINISTIC IN THE OTHER 50 RETURNS?
==============================================================================
  contemporaneous R^2 of ALGO on the other 50: 0.984138
  residual sd 0.001320 vs ALGO sd 0.010482 (12.6% unexplained)
  same-day R^2 out of sample (weights from 1-750): 0.961151
  weights: min +0.0110  max +0.0593  sum +1.0094  negative 0/50
  -> ALGO is a near-exact contemporaneous basket, so the reduction is fair:
     predicting ALGO == predicting the weighted sum of the other 50.

B.  DOES IT SURVIVE ONE STEP AHEAD?
==============================================================================
  ridge replicated from _ridge_forecast_details (lam=1.52), refit daily,
  ALGO added as BOTH predictor and target, scored on returns 750..998
  evaluation days: 249   (binomial SE on accuracy = 0.0317)
                          sign acc  z vs 50%     corr
  ALGO (market proxy)       0.5221      0.70   0.0623
  other 50, mean            0.5235      5.23   0.0745
  other 50, best            0.5904             0.2283
  other 50, worst           0.4378            -0.1252
  -> ALGO ranks 25 of 51 on sign accuracy (24 of the other 50 beat it)

C.  THE AGGREGATION ARGUMENT, MEASURED
==============================================================================
  If ALGO = w . r, then the basket-implied ALGO forecast is w . (per-asset
  ridge forecasts). Compare that against ALGO's realised return.
  basket-implied forecast vs realised basket: corr +0.0763, sign acc 0.5261 (z=+0.82)
  direct ALGO ridge forecast vs realised ALGO: corr +0.0623, sign acc 0.5221

  where the per-asset predictability lives:
  mean per-asset corr, RAW returns            : +0.0745
  mean per-asset corr, CROSS-SECTIONALLY DEMEANED: +0.0772
  -> the ridge's skill is relative-value. Removing the cross-sectional mean
     leaves it intact (or better); aggregating into a basket cancels it.
```
