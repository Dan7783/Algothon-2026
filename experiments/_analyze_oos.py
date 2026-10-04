"""Compare per-pair train/OOS scores and retention: non-windowed latest_tuner vs windowed tuner."""
# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import json
import statistics as st

lt = json.load(open('auto_tuning_results/tuning_results.json'))
wt = json.load(open('auto_tuning_results/windowed_tuning_results.json'))

print('latest_tuner pairs:', len(lt), '| windowed pairs:', len(wt))
print('windowed config:', json.dumps(wt[0]['config']))
print()

hdr = f"{'pair':<14} {'LT train':>9} {'LT OOS':>8} {'LT ret':>7} | {'W train':>8} {'W OOS':>8} {'W ret':>7}"
print(hdr)
print('-' * len(hdr))

lt_ret, w_ret, lt_oos, w_oos = [], [], [], []
for l in lt:
    name = l['pair_name']
    w = next((x for x in wt if x['pair_name'] == name), None)
    lt_tr = l['auto_metrics']['score']
    lt_o = l['test_metrics']['auto']['score']
    r1 = lt_o / lt_tr if lt_tr else float('nan')
    lt_ret.append(r1)
    lt_oos.append(lt_o)
    if w:
        w_tr = w['global']['train_metrics']['score']
        w_o = w['global']['oos_metrics']['score']
        r2 = w_o / w_tr if w_tr else float('nan')
        w_ret.append(r2)
        w_oos.append(w_o)
    else:
        w_tr = w_o = r2 = float('nan')
    print(f"{name:<14} {lt_tr:>9.2f} {lt_o:>8.2f} {r1:>7.1%} | {w_tr:>8.2f} {w_o:>8.2f} {r2:>7.1%}")

print()
print(f"LT retention: mean {st.mean(lt_ret):.1%}  median {st.median(lt_ret):.1%}")
if w_ret:
    print(f"W  retention: mean {st.mean(w_ret):.1%}  median {st.median(w_ret):.1%}")
    print(f"OOS score:    LT mean {st.mean(lt_oos):.2f} | W mean {st.mean(w_oos):.2f}")
    wins = sum(1 for a, b in zip(lt_oos, w_oos) if a > b)
    print(f"LT beats W on OOS in {wins}/{len(lt_oos)} pairs")

# Windowed internals: per-window fragility and selection sources
print()
print('Windowed internals per pair:')
for w in wt:
    fr = sum(1 for x in w['windows'] if x['stability']['fragile'])
    srcs = [x['source'] for x in w['windows']]
    g = w['global']
    print(f"  {w['pair_name']:<14} windows fragile {fr}/4  sources {srcs}  "
          f"global sel={g['selection_source']} fragile={g['fragile']}")
    for x in w['windows']:
        print(f"    w{x['index']} {x['start']}-{x['end']} best {x['best_value']:.2f} "
              f"-> denom {x['denominator']:.2f} [{x['source']}]")
    oc = g.get('objective_comparison') or {}
    for mode, c in oc.items():
        print(f"    [{mode}] trial {c['trial_number']}: ratio_obj {c['ratio_objective']:.3f} "
              f"pct_obj {c['percentile_objective']:.3f} train {c['train_score']:.2f} "
              f"oos {c['oos_score']:.2f}")
