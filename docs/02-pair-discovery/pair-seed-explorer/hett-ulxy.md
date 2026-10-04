# HETT/ULXY robust seed decision

> **Source:** `cusum_output/pair_seed_explorer/hett_ulxy/report.md` in the team's working repository (relocated here for the public release).

## Decision

Only one existing parameter changes:

| Field | Legacy | Selected |
|---|---:|---:|
| `z_entry` | 0.27 | 0.75 |
| `z_exit` | 0.02 | 0.02 |

The higher entry stops HETT/ULXY from entering on small deviations and waits
for a materially larger dislocation. The Kalman/CUSUM filter, ADF and kappa
gates, exit, stop, sizing, and priority remain unchanged.

This filter produced zero CUSUM cuts and both gates blocked zero days in train
and test. `q_beta` values from 0.5 to 3 times the live value generated the same
candidate trades; `h`, `cusum_k`, gate flags, and stops from 3 to 6 were also
operationally flat. There is therefore no evidence for changing beta
responsiveness or relaxing a gate.

## Standalone pair, after 1 bp costs

| Window / metric | Legacy | Selected | Change |
|---|---:|---:|---:|
| Train score | 64.56 | 81.01 | +16.45 |
| Train mean/day | 69.60 | 85.05 | +15.45 |
| Train net | 17,399 | 21,262 | +3,863 |
| Train Sharpe / Sortino | 3.58 / 4.32 | 4.48 / 5.33 | improved |
| Train max drawdown | 1,474 | 1,127 | -23.5% |
| Train CVaR 5% | -525 | -510 | 2.8% less severe |
| Train turnover | 1.208m | 0.970m | -19.7% |
| Test score | 29.05 | 47.24 | +18.19 |
| Test mean/day | 35.70 | 52.11 | +16.42 |
| Test net | 8,925 | 13,029 | +4,104 |
| Test Sharpe / Sortino | 2.09 / 2.35 | 3.11 / 3.54 | improved |
| Test max drawdown | 1,416 | 1,416 | unchanged |
| Test CVaR 5% | -469 | -446 | 4.9% less severe |
| Test turnover | 0.734m | 0.654m | -11.0% |

All five chronological 50-day blocks remain positive. Selected train block
means are `28.56, 97.61, 72.15, 99.16, 127.75`; selected test block means are
`34.34, 83.16, 61.46, 29.61, 52.01`. The legacy blocks were
`23.31, 92.51, 66.10, 71.94, 94.12` and
`24.44, 60.22, 37.83, 29.05, 26.95` respectively.

## Full robust portfolio

The deployed static seed was replayed with independent train/test state, the
current `robust_hybrid_v2` policy, conflicts, position caps, and delayed
eval-style commissions.

| Window / metric | Legacy | Selected | Change |
|---|---:|---:|---:|
| Train score | 791.72 | 806.31 | +14.59 |
| Train mean/day | 802.19 | 816.62 | +14.42 |
| Train Sharpe / Sortino | 8.693 / 10.146 | 8.844 / 10.471 | improved |
| Train max drawdown | 4,986 | 4,986 | unchanged |
| Train CVaR 5% | -2,427 | -2,422 | slightly less severe |
| Train turnover | 66.888m | 66.734m | -0.23% |
| Train half scores | 761.20 / 822.19 | 774.93 / 837.64 | both improve |
| Test score | 830.36 | 846.48 | +16.12 |
| Test mean/day | 839.30 | 855.11 | +15.81 |
| Test Sharpe / Sortino | 9.640 / 12.812 | 9.904 / 13.531 | improved |
| Test max drawdown | 4,060 | 4,060 | unchanged |
| Test CVaR 5% | -1,790 | -1,684 | 5.9% less severe |
| Test turnover | 59.797m | 59.839m | +0.07% |
| Test half scores | 713.93 / 946.65 | 733.58 / 959.26 | both improve |

## Sensitivity and rejected alternatives

The fine screen found an identical train path for entries `0.735-0.765` with
the live `0.02` exit. Full-portfolio validation gave:

| Entry / exit | Train score | Test score | Train max DD |
|---:|---:|---:|---:|
| 0.27 / 0.02 (legacy) | 791.72 | 830.36 | 4,986 |
| 0.735 / 0.02 | 806.31 | 846.17 | 4,986 |
| 0.750 / 0.02 (selected) | 806.31 | 846.48 | 4,986 |
| 0.765 / 0.02 | 806.31 | 849.63 | 4,986 |
| 0.775 / 0.02 | 801.31 | 845.27 | 4,986 |

The rounded `0.75` is selected near the center of the train plateau. `0.765`
has the highest observed test score but sits immediately before the entry-path
cliff at `0.77`; selecting it would be test-window fitting.

- The broad train winner used `0.75/0.10`. Gate removal shown in its printed
  configuration was inert because neither gate blocked a day.
- Exits `0.075-0.10` raised standalone train score to `85.10`, but full test
  score fell to `826.60-828.13`; the live `0.02` exit is retained.
- Exits `0.025-0.030` shared the same pair-level train path and improved the
  standalone test score, but full test score was lower at `844.93`. Adding a
  second parameter change was not justified.
- Higher entries eventually create a fragile train block: at `0.77` the worst
  standalone train block drops from `28.56` to almost zero and drawdown rises
  from `1,127` to `1,618`.

The broad train-ranked search evaluated 63,072 configurations using score,
mean, Sharpe/Sortino, chronological blocks, drawdown, CVaR, and turnover. A
focused 951-point threshold surface then resolved the local trade-path
plateaus. The held-out window informed the final robustness choice, so future
or walk-forward data is now the only clean out-of-sample confirmation.

Re-run the broad search with
`py tune_pair_seed_explorer.py --pair HETT/ULXY`, the fine surface with
`py screen_hett_ulxy_thresholds.py`, and the deployed comparison with
`py validate_hett_ulxy_candidate.py --only legacy selected_live`.
