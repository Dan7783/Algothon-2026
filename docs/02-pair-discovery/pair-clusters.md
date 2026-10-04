# Pair-cluster structure

> **Source:** `analysis_outputs/pair_clusters/report.md` in the team's working repository (relocated here for the public release).

All relationship fitting uses days 1-500. The fitted relation is held
fixed in validation windows 501-625 and 626-750.

## Graph

- Configured edges: **88**
- Assets: **48**
- Assets with degree > 1: **40**
- Exact triangles: **23**
- Candidate triples with at least two edges: **344**
- Connected-component sizes: **[46, 2]**

## Stable multivariate candidates

- Robust triples: **2**
- Robust exact triangles: **0**

```text
       cluster  edge_count anchor   beta_1   beta_2  train_adf_p  val1_adf_p  val2_adf_p  train_mean_return_corr  val1_mean_return_corr  val2_mean_return_corr
EORC-NGTE-SMAH           2   NGTE 0.931693 0.058328     0.000001    0.008198    0.030646                0.221471               0.114002               0.191152
ACAC-HUXZ-NAYO           2   ACAC 0.881804 0.093397     0.000005    0.001208    0.044499                0.139163               0.226820               0.144371
```

## Train-only disjoint triangle selection

Triangles are ranked and made disjoint using days 1-500 only; the validation columns are diagnostics, not selection inputs.

```text
       cluster anchor   beta_1   beta_2  train_adf_p  val1_adf_p  val2_adf_p  validation_adf_pass
ILVX-NGTE-SMAH   SMAH 1.020201 0.046532 7.834992e-08    0.000825    0.013803                 True
AENO-HUXZ-NWIG   AENO 0.152120 0.880378 2.053793e-07    0.071883    0.053419                 True
CTGI-EELT-OTCS   EELT 0.706108 0.046998 4.532838e-06    0.098596    0.025195                 True
```

A connected graph or triangle is not, by itself, evidence of a
tradable cluster. `robust_cluster` additionally requires a fixed
train-fitted basket residual to remain stationary in both
validation halves, positive average return correlation in all
three windows, and Johansen rank of at least one in every window.
