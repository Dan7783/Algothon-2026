# Can per-pair tuning replace the ADF gate?

> **Source:** `cusum_output/daniel_pairs/adf_gate_necessity.md` in the team's working repository (relocated here for the public release).

Answer: **at the pair level yes, at the portfolio level no** — and the
reason is capacity, not trade quality.

## The gate does two jobs

1. **Trade selection** — don't trade a spread that has stopped reverting.
2. **Capacity rationing** — stop the pairs sleeve from occupying the whole
   book, leaving room for ridge and discovery.

Threshold tuning can replace (1). Nothing about a per-pair threshold can
replace (2), because each pair's own optimum is to trade MORE and the cost
lands on OTHER sleeves.

## Evidence 1 — pair level: tuning wins (oracle test)

Each pair, with `require_adf=False`, sweeping z_entry/z_exit/z_stop and
choosing with ORACLE knowledge of both windows (an unachievable upper
bound; the pair's own seeded thresholds are always candidates, so any pair
whose gate never binds recovers exactly by construction):

**20 of 21 pairs recover or beat the gated result.** Only MDGI/HTRK falls
short (-1.06). Large gains: MTNS/ALUT -4.5/0.0 -> 36.9/17.4, MSDP/ULXY
0.0/0.0 -> 24.4/24.7, ACIX/ULXY 0.0/0.0 -> 17.4/27.6, DIHO/RTTH
0.0/-10.1 -> 23.7/6.5, EAFC/SRTX -15.4/0.0 -> -1.2/14.1.

## Evidence 2 — portfolio level: it still loses

| book | gated | un-gated + oracle thresholds |
|---|---|---|
| daniel core | 792.45 / 800.99 | 808.13 (+15.69) / 776.02 (**-24.97**) |
| robust_hybrid | 818.15 / 852.67 | 828.14 (+9.99) / 788.73 (**-63.94**) |

## Evidence 3 — the mechanism, measured

Per-sleeve daily-mean increments, TEST window:

| sleeve | gated | un-gated + oracle | change |
|---|---|---|---|
| atlas | +505.29 | +504.09 | **-1.20** |
| extra lead-lag | +112.27 | +104.75 | -7.52 |
| ridge | +190.46 | +174.71 | -15.75 |
| discovery | +38.04 | +20.25 | -17.79 |

Compare the same decomposition with the gate removed and NO retuning
(`adf_ablation`): atlas test fell **-33.4**. Tuning cut that to **-1.2** —
it solved the trade-selection problem almost perfectly. Every remaining
point of the ~-41 test loss is downstream sleeves being starved of
capacity, exactly the job tuning cannot do.

The same tax applies in train: atlas gains +27.7/day but extraLL (-15.4)
and discovery (-11.5) give most of it back, so the book nets only +3.7.

## Conclusion

Tuning removes ~97% of the gate's trade-selection value (-33.4 -> -1.2)
and 0% of its capacity-rationing value. To drop the ADF gate you must
replace job (2) explicitly — cap concurrent pair positions, cut leg_scale,
or give ridge capacity priority — not tune thresholds harder.

Caveat: the +15.69/-24.97 figures use oracle threshold choice (test data).
Real tuning would do worse; per-pair threshold tuning was separately shown
to be noise-dominated (2,700 configs beat baseline on both windows, zero
survived a one-step robustness nudge).

NOTHING WAS APPLIED. Reproduce: `python research/can_tuning_replace_adf.py`.
