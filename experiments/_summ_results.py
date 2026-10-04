"""Summarise per-window Kalman-ADF screen evidence for the eight core pairs."""
# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import json

import numpy as np

d = json.load(open('auto_tuning_results/screen_train.json'))
recs = d['records']
by_key = {tuple(sorted(r['pair'])): r for r in recs}

watch = [('RTTH', 'NAYO'), ('CTGI', 'EELT'), ('EORC', 'NGTE'), ('HUXZ', 'ACAC'),
         ('HETT', 'ULXY'), ('AENO', 'NWIG'), ('SMAH', 'ILVX'), ('ALUT', 'CCNS')]
print('=== current 8: kalman ADF p per window (chosen orientation) ===')
for p in watch:
    r = by_key[tuple(sorted(p))]
    o = r['orient']
    w = r[o]['windows']
    pk = [round(x.get('adf_p_kalman', 1.0), 3) for x in w]
    ps = [round(x['adf_p'], 3) for x in w]
    hl = [round(x.get('halflife_kalman', float('inf')), 1) for x in w]
    print(f"{p[0]}-{p[1]} ({o}): kalman_p={pk} static_p={ps} hl_k={hl}")

# distribution of kalman-ADF window counts across the universe
def chosen(r):
    return r['A'] if r['orient'] == 'A' else r['B']

cnts = []
for r in recs:
    n = sum(1 for w in chosen(r)['windows'] if w.get('adf_p_kalman', 1.0) < 0.05)
    cnts.append(n)
cnts = np.array(cnts)
print('\nuniverse kalman-adfW counts distribution:')
for k in range(6):
    print(f'  {k}/5: {(cnts == k).sum()} pairs')
print('\nkalman adf p quantiles across all windows of all pairs:')
allp = np.array([w.get('adf_p_kalman', 1.0) for r in recs for w in chosen(r)['windows']])
print(np.round(np.quantile(allp, [0.01, 0.05, 0.1, 0.25, 0.5]), 4))

# how many pairs pass at p<0.10 in >=4/5?
n10 = sum(1 for r in recs
          if sum(1 for w in chosen(r)['windows'] if w.get('adf_p_kalman', 1.0) < 0.10) >= 4)
print(f'\npairs with >=4/5 kalman windows p<0.10: {n10}')
