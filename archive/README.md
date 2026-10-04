# archive/

Superseded strategies, tuners and validation harnesses, kept for provenance. Nothing here is
needed to run the live strategies, and several files depend on modules that were later renamed,
so treat this folder as read-only history.

| Folder | Contents |
|---|---|
| [`legacy/`](legacy/) | Earlier books (`adaptive_four_sigmas`, `robust_hybrid_v2`, `slim_book`, `simple_pairs`, `lean_pairs`, single-pair traders) and the first submission wrappers. |
| [`old_tuners/`](old_tuners/) | The first generation of Optuna / CMA-ES tuners and testbeds (`goat_tuner*`, `phystuner`, `iterative_multi_tuner*`, `single_pair_tuner`) plus the hand-review pair lists. |
| [`pair_validation/`](pair_validation/) | Per-candidate screening and validation harnesses from the seed-explorer phase. |
| [`tuning/`](tuning/) | Pair-agent tuning scripts and their JSON outputs. |
| [`past_submissions/`](past_submissions/) | An early leaderboard submission (20 July). |

## Top-level archived scripts

| Script | What it does |
|---|---|
| [`adaptive_beta_pairs.py`](adaptive_beta_pairs.py) | Pairs-only strategy with smooth beta tracking and CUSUM regime adaptation. |
| [`atlas_signal_gate_diagnostics.py`](atlas_signal_gate_diagnostics.py) | Measure why each fixed atlas pair is inactive under current teamName gates. |
| [`fifty_pairs.py`](fifty_pairs.py) | Fifty labeled pairs: gateless-atlas stable sleeve + gated CUSUM adaptive sleeve. |
| [`four_sigmas_pkg.py`](four_sigmas_pkg.py) | Robust adaptive-pairs hybrid: Daniel core + Four Sigmas discovery. |
| [`four_sigmas_testfit.py`](four_sigmas_testfit.py) | Deliberately test-fitted combiner for the current Four Sigmas signals. |
| [`old_strat.py`](old_strat.py) | Fresh-OOS refactor with a capacity-safe runtime pair-discovery overlay. |
| [`pair_books.py`](pair_books.py) | Two-sleeve pair book: four_sigmas atlas + a broad CUSUM breadth pool. |
| [`poopy_pair_beta.py`](poopy_pair_beta.py) | Isolated pairs strategy with beta speed and CUSUM calibrated in sequence. |
| [`second_adaptive_pairs.py`](second_adaptive_pairs.py) | Isolated pairs strategy with beta speed and CUSUM calibrated in sequence. |
| [`smah_ngte.py`](smah_ngte.py) | SMAH / NGTE single-pair mean-reversion (self-contained). |
| [`spread_regime_pairs.py`](spread_regime_pairs.py) | Atlas-pair research strategy with an explicit spread-regime state machine. |
