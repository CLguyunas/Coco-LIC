# QI-CTDR Stage 2: independent QI observation management

Stage 2 ports the existing QI observation-management work onto the Stage-1
continuous-time LiDAR diagnostic branch. The two modules are deliberately
independent:

- the CT diagnostic always evaluates the complete LiDAR candidate pool before
  QI and remains read-only;
- QI may weight or select LiDAR and visual observations, but it neither reads
  the CT state nor attempts degeneracy recovery;
- later visual weak-direction recovery will consume the CT direction and the
  QI-managed visual candidate pool through a separate fixed-budget module.

This separation preserves the earlier QI work as its own contribution and
makes `original`, `quality-only`, `selection-only`, and `QI` ablations
well-defined.

## Factor-consistent corrections

The port retains the original quality/information idea while correcting four
engineering inconsistencies:

1. A LiDAR line correspondence is represented by the same one-dimensional
   point-to-line distance row used by `LoamFeatureFactorNURBS`; it is not
   replaced by a rank-two vector projection.
2. LiDAR and visual six-dimensional information matrices share one scan-level
   metric. Rotational tangent coordinates are scaled by the median LiDAR range,
   avoiding an implicit comparison between unscaled radians and metres.
3. A quality value `q` is an information multiplier. The residual factor
   receives `sqrt(q)`, so the normal-equation contribution is multiplied by
   `q`. LiDAR quality may attenuate inconsistent residuals (`q < 1`) instead of
   only increasing their influence.
4. The final LIC solve and the marginalization prior receive exactly the same
   selected observations and square-root weights.

Quality statistics are updated from the complete post-solve candidate pool,
not the selected subset. Selection therefore cannot make its own future
residual distribution appear artificially clean.

## Relative information-preservation criterion

The first I-only validation exposed a failure in the original stopping
quantity. It normalized the selected log-determinant gain by the gain from a
small numerical prior. Because the prior log determinant is a large negative
number, roughly 10% of the observations could be reported as retaining 95% of
the information even though the full-sequence position ATE increased by
14.8%.

The selector still ranks candidates by greedy D-optimal marginal gain, but its
stopping quantity is now the dimension-normalized relative D-efficiency

```text
eta_D = exp((log det Lambda_selected - log det Lambda_full) / 6).
```

Both matrices receive the same scale-relative numerical regularizer, so the
ratio is independent of an arbitrary absolute information scale. The
implementation also records the weakest-direction retention

```text
eta_min = lambda_min(
    Lambda_full^(-1/2) Lambda_selected Lambda_full^(-1/2)).
```

`eta_min` is diagnostic only: it introduces no additional threshold or route
into the estimator. Candidate ranking uses an exact lazy-greedy evaluation of
the same D-optimal objective, which preserves the greedy result while avoiding
re-evaluation of every remaining candidate after every selection.

## Configuration

Both master switches are off by default:

```yaml
qi_quality_enable: false
qi_selection_enable: false
qi_lidar_quality_enable: true
qi_visual_quality_enable: true

qi_lidar_q_min: 0.5
qi_lidar_q_max: 1.2
qi_visual_q_min: 0.7
qi_visual_q_max: 1.0
qi_visual_point_quality_weight: 0.25

qi_max_lidar_obs: 800
qi_max_visual_obs: 200
qi_selection_d_efficiency: 0.95
qi_selection_min_gain: 1.0e-6
qi_info_prior_eps: 1.0e-6
qi_output_csv: true
qi_log_enable: false
```

`qi_selection_info_ratio` remains accepted as a legacy alias so an existing
local YAML does not silently fall back to a default. New configurations should
use `qi_selection_d_efficiency`.

The output file is:

```text
data/<bag-name>_qi_observation.csv
```

New CSVs identify the metric explicitly with
`lidar_d_efficiency`, `visual_d_efficiency`,
`lidar_min_direction_retention`, and
`visual_min_direction_retention`. The analyzer still reads historical
`*_info_coverage` files, but labels them as
`legacy_logdet_gain_ratio`; it never presents those historical values as the
corrected D-efficiency.

## Required ablation

Use identical base Coco-LIC parameters for all four runs:

| Run | `qi_quality_enable` | `qi_selection_enable` | Purpose |
|---|---:|---:|---|
| Original | false | false | untouched Coco-LIC path |
| Q | true | false | quality weighting only |
| I | false | true | information selection only |
| QI | true | true | complete existing QI work |

Do not enable CT-driven visual recovery in this experiment. Stage 2 must first
show that QI has internally consistent weights, respects its observation
budget, retains the requested relative D-efficiency where the budget permits,
does not collapse weakest-direction retention, and does not create a
trajectory failure.

Audit one or more QI runs with:

```bash
python3 tools/analyze_qi_observation.py \
  --run q=data/<bag>_Q_qi_observation.csv \
  --run i=data/<bag>_I_qi_observation.csv \
  --run qi=data/<bag>_QI_qi_observation.csv
```

Trajectory accuracy and runtime are reported separately. The internal CSV is
evidence for weight semantics, selected/candidate ratios, relative
D-efficiency, weakest-direction retention, and pre/post residual behaviour;
it is not a substitute for trajectory evaluation.
