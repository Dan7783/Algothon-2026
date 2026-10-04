# Generator forensics: what process produced prices.txt?

> **Source:** `analysis_outputs/generator_forensics/forensics.txt` in the team's working repository (relocated here for the public release).

```text
prices.txt: 51 instruments x 1000 days
==============================================================================

[1] PRICE QUANTISATION
  every price exactly 2dp: True
  min price 5.07   max price 809.85
  median rounding noise / typical daily move: 0.0072
  -> rounding is small vs daily moves, so it does NOT hide the dynamics,
     but it does destroy exact float reproduction of any candidate stream.

[2] RETURN DISTRIBUTION  (is it plain Gaussian?)
  skew      mean -0.007  min -0.154  max +0.149
  exkurt    mean -0.030  min -0.308  max +0.416
  Jarque-Bera rejected at 5%: 1/51 assets
  Shapiro-Wilk rejected at 5%: 1/51 assets
  (at 5% you expect ~2.6/51 rejections by chance)

[3] VOLATILITY DYNAMICS  (constant sigma, or GARCH-like?)
  ARCH (Ljung-Box on squared z, 10 lags) rejected at 5%: 6/51 assets
  2nd-half / 1st-half vol ratio: mean 1.005 [0.886, 1.154]

[4] RETURN AUTOCORRELATION  (is each asset a clean random walk?)
  Ljung-Box on returns rejected at 5%: 4/51 assets
  lag-1 autocorrelation: mean +0.0101  range [-0.1099, +0.1199]
  |lag-1| > 2SE (0.0633): 9/51 assets

[5] FACTOR STRUCTURE  (Marchenko-Pastur: how many real factors?)
  q = N/T = 0.051   MP upper edge = 1.503
  eigenvalues above MP edge: 3
  top 8: 11.89  2.76  1.94  1.20  1.18  1.05  1.04  1.01
  top eigenvalue explains 23.3% of variance

[6] LATTICE / DISCRETENESS  (does a raw RNG stream show through?)
  distinct log returns: 46,618 of 50,949
  smallest gap between distinct returns: 4.068e-09
  -> log returns inherit price rounding, so they sit on a price-dependent
     lattice; there is no single global tick to invert.

[7] IS THE COINTEGRATION EXACT?  (imposed by construction, or estimated?)
  For a hand-built generator, an imposed pair has a beta that is EXACTLY
  constant across subsamples. Drift in beta means it was not imposed.
  most mean-reverting pairs (rho = residual AR(1); drift = |b2-b1|/|b|):
    AENO-NWIG   rho=0.8523  half-life=  4.3d  beta drift=  8.97%
    SMAH-ILVX   rho=0.8611  half-life=  4.6d  beta drift=  2.11%
    EORC-NGTE   rho=0.8849  half-life=  5.7d  beta drift= 12.27%
    HUXZ-ACAC   rho=0.8979  half-life=  6.4d  beta drift= 29.54%
    HETT-ULXY   rho=0.8984  half-life=  6.5d  beta drift= 29.27%
    MHRM-EAFC   rho=0.9166  half-life=  8.0d  beta drift=  2.39%
    CTGI-EELT   rho=0.9180  half-life=  8.1d  beta drift=  5.12%
    RTTH-NAYO   rho=0.9193  half-life=  8.2d  beta drift=  4.56%
  median beta drift across all 1,275 pairs: 153.65%
```
