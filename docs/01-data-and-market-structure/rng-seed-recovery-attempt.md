# RNG seed-recovery attempt

> **Source:** `analysis_outputs/generator_forensics/seed_search.txt` in the team's working repository (relocated here for the public release).

```text
SEED RECOVERY ATTEMPT
==============================================================================
search: 100,000 seeds x 2 bit generators x 2 draw orderings

generating 100,000 candidate streams from PCG64 ...
  H1 asset time-series     k= 64 tests=5,100,000  best |r|=0.5644 at seed 87537, target 37
  H2 daily cross-section   k= 51 tests=6,400,000  best |r|=0.6836 at seed 60458, target 9
  H3 common factors        k= 64 tests=  300,000  best |r|=0.5817 at seed 48515, target 1

generating 100,000 candidate streams from MT19937 ...
  H1 asset time-series     k= 64 tests=5,100,000  best |r|=0.5976 at seed 91444, target 23
  H2 daily cross-section   k= 51 tests=6,400,000  best |r|=0.6864 at seed 29289, target 36
  H3 common factors        k= 64 tests=  300,000  best |r|=0.5527 at seed 7677, target 1

==============================================================================
CALIBRATION: what would pure noise have produced?
  total hypotheses tested: 23,600,000
  hypothesis                  k      tests  observed  noise med  noise p99
  H1 asset time-series       64  5,100,000    0.5644     0.6028     0.6656
  H2 daily cross-section     51  6,400,000    0.6836     0.6640     0.7267
  H3 common factors          64    300,000    0.5817     0.5516     0.6252
  H1 asset time-series       64  5,100,000    0.5976     0.6028     0.6656
  H2 daily cross-section     51  6,400,000    0.6864     0.6640     0.7267
  H3 common factors          64    300,000    0.5527     0.5516     0.6252
  -> 0 of 6 searches beat their own noise ceiling

==============================================================================
EXTENSION TEST: re-score each winner on 512 draws instead of 64
  a real seed match is invariant to length; a lucky one collapses to ~0
  (H2 has fixed width 51, so it is extended by asking whether the NEXT
   block of 51 draws matches the NEXT day's cross-section)
  PCG64    H1 asset time-series     seed  87537  |r|@64 = 0.5644  ->  |r|@ext = 0.0391
  PCG64    H2 daily cross-section   seed  60458  |r|@51 = 0.6836  ->  |r|@ext = 0.2868
  PCG64    H3 common factors        seed  48515  |r|@64 = 0.5817  ->  |r|@ext = 0.0606
  MT19937  H1 asset time-series     seed  91444  |r|@64 = 0.5976  ->  |r|@ext = 0.0785
  MT19937  H2 daily cross-section   seed  29289  |r|@51 = 0.6864  ->  |r|@ext = 0.0527
  MT19937  H3 common factors        seed   7677  |r|@64 = 0.5527  ->  |r|@ext = 0.0369

  candidates surviving the extension test: 0

==============================================================================
WHY STATE RECOVERY CANNOT WORK HERE, EVEN IN PRINCIPLE
  prices are quoted to 2dp, so each log return is known to about 6.7 bits (401 returns are exactly 0)
  MT19937 state recovery needs 624 EXACT 32-bit outputs = 19,968 bits,
  with zero tolerance for error. PCG64 needs its 128-bit state exactly.
  Observed returns are also a MIXTURE (3 factors + idiosyncratic + OU),
  so no single observation is one raw draw: you would have to invert the
  mixture with estimated loadings first, adding error far larger than the
  rounding itself. The map from data back to raw draws is not invertible.
```
