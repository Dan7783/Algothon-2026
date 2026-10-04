# Multi-Signal Overfitting — Reference Brief

> **Source:** `novy-marx-multisignal-brief.md` in the team's working repository (relocated here for the public release).

Source: Robert Novy-Marx, "Backtesting Strategies Based on Multiple Signals",
NBER Working Paper 21329 (July 2015). Restated in full; no PDF needed.

---

## 1. The claim

Combining multiple signals biases the backtested t-statistic upward even when
every signal is pure noise, because **signals are almost always signed so that
each predicts positive in-sample returns**. That sign flip uses full-sample
information. It is data snooping, present even with zero selection bias.

Two distinct biases:

- **Selection bias (multiple testing)** — tried n, reported the best.
  Bonferroni-ish, well understood.
- **Overfitting bias** — used all n, but signed each to work in sample.
  Present even with no selection.

They compound multiplicatively. Headline: **combining the best k of n
candidate signals is about as biased as picking the single best of n^k.**

At the 5% level: best 3-of-10 → t ≈ 4; best 3-of-20 → t ≈ 5; best 7-of-100 →
t ≈ 7; best 1-of-1 → t ≈ 2.1 (not 1.96 — real return series have fat tails).
A backtest with t = 5 built from 10 candidate signals is not evidence.

## 2. The model

n candidate signals; strategy i holds stock j in proportion to
`(S_jt − median_t(S)) · m_jt`. Individual-strategy returns are assumed normal,
**uncorrelated across signals**, equal volatility. Because weights are linear
in the signal, a strategy trading a composite signal is *exactly* a portfolio
of the individual-signal strategies — "integrated" and "siloed" coincide.

With `t = (t_1..t_n)` the individual t-stats and weights ω:

```
t_ω = ω'·t / sqrt(ω'·ω)                                    (2)
```

| Weighting | ω | Portfolio |
|---|---|---|
| Equal-weight signals | `1/‖1‖₁` | minimum variance (MV) |
| Signal-weight signals | `t/‖t‖₁` | ex-post mean-variance efficient (MVE) |

Signal weighting is the more common "sophisticated" choice and is **strictly
worse** for overfitting — more freedom to overweight whatever worked.

## 3. Core algebra

Ordered `t_(1) ≤ ... ≤ t_(n)`, best k of n:

```
t^MV_{n,k}  = ( Σ_{i=1..k} t_(n+1−i) ) / sqrt(k)          (3)
t^MVE_{n,k} = sqrt( Σ_{i=1..k} t²_(n+1−i) )               (4)
```

- `t^MV = √k × (mean t of signals employed)`. **k signals each with t ≈ 1
  give combined t ≈ √k.** Five junk signals at t = 1 → t ≈ 2.2, passes a
  naive 1.96 test.
- `t^MV ≤ t^MVE` always, equality only if the top-k order stats are identical.

## 4. Critical values

### 4.1 Best 1-of-n (pure selection)
```
t*_{n,1,p} = N⁻¹( (1 − p/2)^{1/n} )                       (5)
```
Practical rule: `p_true ≈ n · 2·N(−|t|)`.

### 4.2 Best n-of-n, signal-weighted (pure overfitting, MVE)
```
t^MVE_{n,n} = ‖t‖₂ ~ sqrt(χ²_n)                           (6)
t**_{n,n,p} = sqrt( Φ⁻¹_{χ²_n}(1 − p) )                   (7)
```

### 4.3 Best n-of-n, equal-weighted (pure overfitting, MV)
```
t^MV_{n,n} → sqrt(2n/π) + sqrt(1 − 2/π)·χ                 (8)
t*_{n,n,p} ≈ sqrt(2/π)·√n + sqrt(1 − 2/π)·N⁻¹(1 − p)      (9)
t*_{n,n,p} ≈ sqrt(2/π)·√n + sqrt(1 − 2/π)·N⁻¹(1 − np/(n+1))  (10)
```
(9) is a lower bound — true distribution is right-skewed. (10) corrects it.

### 4.4 General best k-of-n

Exact via the order-statistic simplex: order stats of n standard uniforms are
uniform on `Δⁿ = {0 ≤ u_1 ≤ ... ≤ u_n ≤ 1}`. Pull back with
`t = N⁻¹((1 + u)/2)`:

```
P[t^MV_{n,k}  > τ] = E[ 1{ ‖I_{n,k} t‖₁ ≥ √k·τ } ]        (11)
P[t^MVE_{n,k} > τ] = E[ 1{ ‖I_{n,k} t‖₂ ≥ τ } ]           (12)
```

**Analytic approximations** (Appendix D):

```
μ_{n,k} = N⁻¹( 1 − (k+1)/(2(n+1)) )
λ_{n,k} = n(μ_{n,k}) / (1 − N(μ_{n,k})) = 2·((n+1)/(k+1))·n(μ_{n,k})
Σ_{n,k} = 1 + μ_{n,k}·λ_{n,k}

μ_MV    = √k · λ_{n,k}
σ²_MV   = Σ_{n,k} − λ²_{n,k} + k(n−k)(λ_{n,k} − μ_{n,k})² / ((k+1)(n+2))

μ_MVE2  = k · Σ_{n,k}
σ²_MVE2 = k(μ³_{n,k}·λ_{n,k} + 3Σ_{n,k} − Σ²_{n,k})
          + k²(n−k)(Σ_{n,k} − μ²_{n,k})² / ((k+1)(n+2))

t*_{n,k,p}  ≈ μ_MV + σ_MV · N⁻¹(1 − kp/(k+1))            (13)
t**_{n,k,p} ≈ sqrt( μ_MVE2 + σ_MVE2 · N⁻¹(1 − kp/(k+1)) ) (14)
```

**Typo note:** the paper's eq. (37) prints `μ_MVE2 = k²(1 + μλ)`, contradicting
the body text and the derivation at (30)–(31). **Use `k·Σ_{n,k}`.**

Variance terms are insensitive to n and k; the **means** drive the movement.
The null shifts right, it does not fatten much.

## 5. Optimal k ≈ n/2

More signals → lower combined vol (helps) but worse average quality (hurts).
`d/dk E[t^MV_{n,k}] = 0` reduces to `E[t_(n−k)] = E[λ(t_(n−k))]/2`: **stop
adding signals when the next is only half as good as the average of those you
have.** With `x* = 0.612` (root of `2x = n(x)/N(−x)`):

```
(k+1)/(n+1) ≈ 2·N(−x*) = 0.541    →    k ≈ n/2            (17)
```

Diagnostic: a published multi-signal strategy using *no* weak signals is
itself evidence of discarded candidates, i.e. selection bias on top of
overfitting bias. Then:

```
E[t^MV_{2k,k}] ≈ 4n(N⁻¹(3/4))·√k = 1.27√k     (with selection)
E[t^MV_{k,k}]  ≈ sqrt(2k/π)      = 0.80√k     (without)
```
— ~60% higher under selection.

## 6. The power law

```
n* = ln(1 − p/2) / ln(N(τ))                               (18)
```

**Panel A — equal-weighted (MV):**

| n \ k | 2 | 3 | 4 | 5 | 10 |
|---|---|---|---|---|---|
| 10 | 57 | 164 | 305 | 426 | 141 |
| 20 | 227 | 1,250 | 4,490 | 12,000 | 113,000 |
| 50 | 1,360 | 17,500 | 147,000 | 900,000 | 5.03×10⁸ |
| 100 | 5,220 | 128,000 | 2.02×10⁶ | 2.33×10⁷ | 2.85×10¹¹ |

**Panel B — signal-weighted (MVE):**

| n \ k | 2 | 3 | 4 | 5 | 10 |
|---|---|---|---|---|---|
| 10 | 73 | 260 | 614 | 1,102 | 2,690 |
| 20 | 277 | 1,810 | 7,680 | 24,400 | 658,000 |
| 50 | 1,600 | 23,400 | 223,000 | 1.55×10⁶ | 1.70×10⁹ |
| 100 | 6,020 | 165,000 | 2.88×10⁶ | 3.70×10⁷ | 7.78×10¹¹ |

Panel A is non-monotone at n=10 (k=10 < k=5) — the k ≈ n/2 peak. Panel B is
monotone (adding assets weakly improves the ex-post MVE frontier).

`ln n*` ~ `k·λ²_{n,k}`, approximately affine in `ln n` with slope ∝ k, hence
**n* ∝ n^k**. Intuition: best k-of-n is worse than splitting n into k random
subsamples of size n/k and taking the best from each; bias is highly convex in
candidate count, and the effects multiply across the k picks.

## 7. Robustness

- Quantile sorting, rank weighting, z-score weighting are indistinguishable
  (R² 99.4% between rank- and equal-weighted tertile VMG). Equal- vs
  value-weighting the capitalization multiplier *does* matter.
- Theoretical critical values sit uniformly **~7% below** empirical
  (bootstrapped-from-real-returns) ones — the model lacks excess kurtosis and
  heteroskedasticity. Treat analytic numbers as a **floor**.

## 8. Prescription

- **Evaluate the marginal power of each signal individually.** Main
  recommendation.
- Do combine signals you believe in on independent grounds — Markowitz holds.
- Do **not** believe in a set of signals because the combination backtests
  well. Backtesting well together implies nothing about any of them.
