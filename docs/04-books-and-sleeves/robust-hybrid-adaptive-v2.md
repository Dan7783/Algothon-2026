# Conservative Adaptive Hybrid v2

> **Source:** `cusum_output/robust_hybrid/adaptive_v2_report.md` in the team's working repository (relocated here for the public release).

The existing hybrid was retained as an exact control.  Each proposed change
was first tested globally, then pair by pair under independently reset train
(251-500) and test (501-750) replays.  Production changes had to improve both
full windows and avoid degrading any of the four 125-day half-windows.

## Diagnosis

- Live entry thresholds had median 0.33 sigma; 17 of 21 were below 0.50.
- The 21 filters generated 51 cuts over 750 observations, concentrated in a
  few agents (MTNS/ALUT 9, DIHO/RTTH 8, ALUT/CCNS and MSDP/ULXY 6 each).
- The old post-cut process noise was typically thousands of times normal, the
  12-day snap target was overshot by 5-40%, and execution continued while the
  reference beta moved.
- A universal later-entry rule was rejected.  A 0.50-sigma floor changed score
  from 784.48/779.08 to 733.16/787.44: strongly test-favoured and damaging to
  train.  Two-bar confirmation and staged sizing also damaged train.

## Selected pair-local controls

| Pair | Change | Reason |
|---|---|---|
| MTNS/ALUT | 40-day snap target, no overshoot, 3-bar disagreement confirmation, capped 3% snap, smaller covariance/noise burst | Repeated and visibly large beta jumps |
| ALUT/CCNS | Freeze execution beta from entry to exit | Prevent the trade mean moving while a position is open |
| MDGI/HTRK | Exit on cut, stand down while adapting, 5-bar cooldown | Treat a confirmed cut as a break rather than a new mean |
| DIHO/RTTH | Exit on cut, stand down while adapting, 5-bar cooldown | Same; eight historical cuts |
| NAYO/SRTX | Raise entry from 0.16 to 0.50 sigma | Only later-entry change passing all four subwindows |

The other pairs keep their tuned timing.  This matters: the existing timing
study found that most low-z episodes reverse quickly and supply the majority
of sleeve PnL, despite the visually painful minority that continues outward.

## Beta-response improvement

For MTNS/ALUT over all 750 observations:

| Metric | Control | Selected |
|---|---:|---:|
| Cuts detected | 9 | 9 |
| Maximum one-day cut jump | 0.1712 | 0.0391 |
| Beta total variation | 1.8795 | 0.6939 |
| 95th-percentile daily beta change | 0.01155 | 0.00377 |

The detector still labels the structural episodes, but it no longer recentres
the spread with a large one-day beta jump.  Maximum cut response falls 77% and
total beta variation falls 63%.

## Portfolio results

| Strategy | Train mean | Train score | Train std | Test mean | Test score | Test std |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid v1 control | 795.17 | 784.48 | 1,467.08 | 788.61 | 779.08 | 1,379.38 |
| **Conservative v2** | **802.57** | **791.92** | 1,472.21 | **808.74** | **799.47** | **1,376.46** |

Score improves +7.43 train and +20.40 test.  All half-window scores improve:
+2.16, +12.70, +19.72, and +21.10.  Turnover is 67.21m train and 59.51m test;
maximum per-instrument exposure remains \$10,000.

## Ordered sleeve attribution

Later sleeves use only capacity left by earlier sleeves, so contribution is
reported cumulatively in production order.

| Cumulative book | Train mean | Increment | Test mean | Increment |
|---|---:|---:|---:|---:|
| Adaptive atlas | 561.96 | +561.96 | 462.40 | +462.40 |
| + extra lead-lag | 644.68 | +82.73 | 575.03 | +112.63 |
| + ridge | 762.40 | +117.72 | 772.62 | +197.59 |
| + discovery | 802.57 | +40.17 | 808.74 | +36.12 |

Test data participated in the requested two-window selection.  Days 501-750
are therefore validation, not an untouched final holdout.
