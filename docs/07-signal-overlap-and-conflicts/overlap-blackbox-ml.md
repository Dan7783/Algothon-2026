# Current signal-overlap black-box

> **Source:** `analysis_outputs/current_overlap_blackbox/report.md` in the team's working repository (relocated here for the public release).

This experiment contains no pair name, instrument identity, or protected-pair
indicator. Days 751-1000 are unavailable during model and threshold
selection.

## Development result

- Accepted: **False**
- Model: `gbm_stumps`
- Router mode: `pooled`
- Confidence margin: `0.15`
- Train AUC: `0.5692`
- Validation AUC: `0.4825`
- VAL1 score delta: `0.0000`
- VAL2 score delta: `0.0000`
- Full validation score delta: `0.0000`

The production strategy is not changed unless both validation halves
improve and the locked OOS result also supports integration.

## Leading development candidates

```text
                model router_mode  confidence_margin  train_auc  val1_auc  val2_auc  val1_delta  val2_delta  validation_delta  change_rate
           gbm_stumps      pooled               0.15   0.569213  0.492294  0.473157    0.000000    0.000000          0.000000     0.000000
random_forest_shallow      pooled               0.15   0.593220  0.496104  0.468613    0.000000    0.000000          0.000000     0.000000
  extra_trees_shallow      pooled               0.15   0.563798  0.516292  0.471903    0.000000    0.000000          0.000000     0.000000
  extra_trees_shallow   segmented               0.15   0.580222  0.524754  0.475964    0.000000    0.000000          0.000000     0.000000
random_forest_shallow   segmented               0.15   0.617682  0.503527  0.468825   -6.688401    0.000000         -3.338423     0.000265
           gbm_stumps      pooled               0.10   0.569213  0.492294  0.473157  -10.384598  -15.176996        -12.744507     0.007951
           gbm_stumps   segmented               0.15   0.596983  0.515262  0.468572   -5.715291  -16.094464        -10.908946     0.003711
          logistic_l2      pooled               0.15   0.550339  0.503914  0.485458  -14.447323  -20.928089        -17.656262     0.007156
  extra_trees_shallow      pooled               0.10   0.563798  0.516292  0.471903  -19.738738  -26.830028        -23.275733     0.005566
          logistic_l2      pooled               0.10   0.550339  0.503914  0.485458   50.082305  -32.183893          8.997605     0.040551
```
