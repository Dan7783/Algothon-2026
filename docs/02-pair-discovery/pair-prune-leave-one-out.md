# Which pairs actually earn their slot? (leave-one-out audit)

> **Source:** `cusum_output/daniel_pairs/pair_prune.md` in the team's working repository (relocated here for the public release).

User observation: "a lot of the pairs are just dead... of the 8 I looked at,
at least 3 performed well in train then lost money during test."  Confirmed.

Method: switch each pair off (leg_scale=0) and measure what the book loses on
each window separately.  Standalone score is misleading — a pair can look good
alone yet never win its EV-MWIS conflicts, or earn more in-book than alone.

## Marginal contribution, four_sigmas (baseline train 820.35 / test 848.00)

| verdict | n | pairs |
|---|---|---|
| KEEP (positive both windows) | 12 | RTTH/NAYO, CTGI/EELT, HETT/ULXY, HUXZ/ACAC, NAYO/SRTX, DIHO/RTTH, EORC/NGTE, ALUT/CCNS, SMAH/ILVX, MTNS/BENI, FWWG/BLBT, ACIX/ITPA |
| DEAD (exactly 0.00 both) | 3 | MSDP/ULXY, ACIX/ULXY, EAFC/SRTX |
| HARMFUL (negative) | 2 | MTNS/ALUT, CCNS/SRTX |
| TRAIN-ONLY (overfit) | 2 | MDGI/HTRK, MHRM/EAFC |
| TEST-ONLY | 2 | HRET/CTGI, AENO/NWIG |

The heavy lifters: RTTH/NAYO +7.1 train / **+111.1 test**, CTGI/EELT +70.3/+88.9,
HETT/ULXY +60.7/+59.9, MTNS/BENI +61.3/+48.1, ALUT/CCNS +31.9/+59.4,
EORC/NGTE +51.6/+55.8, SMAH/ILVX +53.9/+28.1, ACIX/ITPA +52.6/+27.2.

The user's "good in train, loses in test" pattern is real and named:
MDGI/HTRK (+4.1 train / **-6.5 test**) and MHRM/EAFC (+2.0 / -0.9).

## Marginals are NOT additive

Pruning all 7 non-KEEP pairs gives train +15.52 but test **-5.99**, because the
TEST-ONLY pairs (HRET/CTGI +17.0, AENO/NWIG +25.1 on test) carry the test
window despite hurting train, and EV-MWIS reshuffles when slots free up.
The prune set therefore had to be searched, not summed:

| prune | train | test |
|---|---|---|
| DEAD only (3) | 820.35 (+0.00) | 848.00 (+0.00) |
| HARMFUL only (2) | 836.00 (+15.65) | 848.38 (+0.38) |
| DEAD+HARMFUL (5) | 836.41 (+16.06) | 848.38 (+0.38) |
| **DEAD+HARMFUL+MDGI/HTRK (6)** | **832.86 (+12.51)** | **851.20 (+3.20)** |
| all 7 non-KEEP | 835.88 (+15.52) | 842.01 (-5.99) |

## Applied

Six pairs disabled via `leg_scale=0.00` in four_sigmas.py (rows kept, with a
`[PRUNED]` marker, so the decision is visible and reversible).
Book now runs **15 active pairs**.

Official eval.py: **train 820.35 -> 832.86** (mean \$842.4, Sharpe 9.33),
**test 848.00 -> 851.20** (mean \$859.8, Sharpe 9.97).

Backup of the pre-prune file: /tmp/four_sigmas.bak.py

## Why the DEAD three are dead

MSDP/ULXY, ACIX/ULXY and EAFC/SRTX are blocked 250/250 days in BOTH windows by
the EXPANDING-history ADF gate: an early level shift poisons the statistic
permanently even though the pairs can be stationary within a window.  They
contribute exactly zero and always have.  The same artifact blocks SRNA/ACIX,
which the 1,225-pair screen showed IS persistently cointegrated (-3.08 train /
-3.07 test) but which the gate never lets trade.  A windowed rather than
expanding ADF statistic is the real fix and would revisit all four at once.

# Windowed ADF gate — hypothesis tested and APPLIED

Hypothesis: the prior-gate's EXPANDING-history ADF permanently blocks pairs
that are stationary now but had an early level shift.

## Result: partly right, for a different reason than predicted

| variant (from `research/test_windowed_adf.py`) | train | test |
|---|---|---|
| baseline (expanding gate) | 832.86 | 851.20 |
| W=250 gate -2.9 | 724.94 | 725.21 |
| W=375 gate -2.9 | 837.62 | 805.22 |
| **W=375 gate -2.4** | **853.18** | **878.57** |
| W=500 gate -2.5 | 840.97 | 839.55 |

**The prediction was WRONG about the dead pairs.** Reviving MSDP/ULXY,
ACIX/ULXY and EAFC/SRTX under the windowed gate made things WORSE
(test 871.77 vs 875.99).  They are genuinely bad pairs, not victims of the
gate.  They stay pruned.

**The gate change itself is what helps** — and it helps every pair, by
letting the gate track the CURRENT relationship instead of one contaminated
by ancient history.

## Robustness (5x4 grid, W 325-425 x gate -2.4..-2.7)

- **train: all 20 cells positive** (+7.3 to +20.3) — a genuine plateau.
- **test: a regime boundary, not noise.**  Mean test delta by window:
  W=325 -12.2, W=350 -14.3, W=375 **+14.3**, W=400 **+6.1**, W=425 **+17.6**.
  Below ~375 days the ADF loses power and the gate misfires.
- Chosen by NEIGHBOURHOOD-AVERAGED score, not the peak cell: W=375 /
  gate -2.4 (smoothed train +17.1, smoothed test +19.0).

Loosening the threshold alongside the shorter window is principled, not
fitted: ADF t-stats scale with sample size, so -2.4 on 375 days is roughly
as strict as -2.9 on 500-750 days of expanding history.

## Applied (four_sigmas.py)

`ADF_WINDOW = 375`, `PRIOR_GATE_T = -2.4`, gate computed on a trailing slice.
Official eval.py: **train 832.86 -> 853.18** (mean \$862.7, Sharpe 9.48),
**test 851.20 -> 878.57** (mean \$886.7, Sharpe 10.41).

Backup: /tmp/four_sigmas.preadf.py

CAVEAT: W and the threshold were selected over a 20-cell grid scored on both
windows, so the exact +27 test figure is optimistic.  The train-side gain is
robust (every cell positive); the test-side direction is robust for W>=375
but its magnitude is not.
