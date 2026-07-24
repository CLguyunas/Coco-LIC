# QI-CTDR Stage 3: continuous-time weak-direction visual complement

Stage 3 implements the recovery link agreed after the Stage-1 detector and
Stage-2 QI ablation had been validated. It does not replace the existing QI
work and does not reinterpret QI as a degeneracy detector.

The engineering boundary is:

- Stage 1 detects persistent LiDAR geometric weakness from the complete
  LiDAR candidate pool using existing continuous-time LiDAR factor Jacobians.
- Stage 2 independently produces a quality-aware, information-preserving
  baseline subset of existing LiDAR and visual observations.
- Stage 3 is activated only by a persistent Stage-1 LiDAR weakness. It searches
  only the real tracked visual observations omitted from the Stage-2 baseline
  and adds those that complement the measured weak directions.

No IMU propagation factor, weak-direction pseudo-measurement, subspace prior,
CASR residual, or spline-support-degradation route is introduced.

## Existing Coco-LIC quantities

For an image at time `t`, Coco-LIC already represents every tracked 3D--2D
correspondence with `PnPFactorNURBS`. The factor directly depends on the four
active non-uniform spline control rotations and positions and is protected by
the existing Cauchy loss with scale 10.

Let `delta x` contain the free spline-control perturbations used by the
Stage-1 LiDAR information matrix. Let

```text
delta xi_s(t) = A_s(t) delta x
```

be the exact NURBS mapping to the characteristic-length-scaled six-dimensional
reference-pose tangent. `A_s` is derived from the same non-uniform blending
matrices and the same control points as the current estimator.

Stage 1 supplies a matrix `W` whose orthonormal columns are the weak
eigen-directions of the LiDAR-only reference-pose information. This is an
existing output of the Stage-1 method; Stage 3 does not estimate a second,
independent weak basis.

## New Stage-3 factor-consistent projection

For a real visual observation `i`, Stage 3 evaluates the analytic Jacobian
`J_k,i` of the existing `PnPFactorNURBS` with respect to the same control-point
state. Its Jacobian in the Stage-1 scaled reference-pose tangent is

```text
J_p,i = J_k,i A_s(t)^+ ,
```

where `A_s(t)^+` is the minimum-norm right inverse obtained from a full-row-rank
SVD. This is not a new observation model. It is a coordinate transformation
used only to compare an existing visual factor with the LiDAR weak basis.

Each candidate must pass the chain-rule consistency audit

```text
epsilon_i =
  ||J_k,i - J_p,i A_s(t)||_F / max(||J_k,i||_F, epsilon) <= 1e-5.
```

A candidate that fails this audit is excluded rather than approximated. The
information matrix

```text
H_i = J_p,i^T J_p,i
```

uses the observation's existing QI square-root weight and the local first
derivative of Coco-LIC's existing Cauchy loss. Stage 3 introduces no new
factor-strength parameter.

## New weak-subspace complement criterion

Let `B` be the QI-selected visual baseline and `F` the full factor-consistent
visual candidate set. Their information in the LiDAR weak subspace is

```text
G(S) = Lambda_W + W^T (sum_{i in S} H_i) W,
```

where `Lambda_W` is the same scale-relative numerical regularizer already used
by Stage-2 information selection. The dimension-normalized weak-subspace
D-efficiency is

```text
eta_W(S) =
  exp((log det G(S) - log det G(F)) / rank(W)).
```

Starting from `B`, Stage 3 greedily adds an omitted real visual observation
with the largest marginal increase in `log det G`. Selection stops when:

1. `eta_W` reaches the existing QI D-efficiency target;
2. the existing QI minimum marginal-gain condition is reached;
3. no factor-consistent candidate remains; or
4. the fixed additional-observation budget is exhausted.

The selector only augments `B`; it never removes a QI-selected observation.
Consequently, its unregularized positive-semidefinite visual information
cannot decrease. The CSV nevertheless records both weak-subspace and global
D-efficiency before and after augmentation so that this invariant can be
checked from every run.

## Estimator write path

With `apply_to_estimator: false`, Stage 3 is a shadow audit and changes no
trajectory variable.

With `apply_to_estimator: true`, the selected observations are appended to the
existing final-iteration PnP arrays. `TrajectoryManager` then:

1. adds the unchanged `PnPFactorNURBS` factors to the final LIC solve;
2. caches the same points, pixels, and square-root weights; and
3. inserts those same factors into the matching marginalization prior.

Thus the write path adds only genuine visual evidence already tracked by
Coco-LIC. The weak direction controls observation choice but never becomes a
constraint by itself.

## Configuration

Stage 3 requires `lidar_iter >= 2`, the Stage-1 detector, and Stage-2
information selection:

```yaml
ct_degeneracy:
  enabled: true
  apply: false
  output_csv: true
  weak_eigenvalue_ratio: 3.0e-3

qi_selection_enable: true
qi_selection_d_efficiency: 0.95
qi_selection_min_gain: 1.0e-6
qi_info_prior_eps: 1.0e-6

ct_visual_recovery:
  enabled: true
  apply_to_estimator: false
  output_csv: true
  max_additional_visual_observations: 64
```

The D-efficiency target, numerical regularizer, and marginal-gain threshold
are reused from QI. The only new Stage-3 method hyperparameter is the maximum
number of additional real visual observations; it is a computational budget,
not a residual weight.

The output is:

```text
data/<bag-name>_ct_visual_recovery.csv
```

Audit it with:

```bash
python3 tools/analyze_ct_visual_recovery.py \
  --run shadow=data/<bag-name>_ct_visual_recovery.csv
```

An analyzer `passed=True` result certifies structural invariants such as
factor accounting, budget compliance, non-decreasing information metrics, and
chain-rule consistency. It does not by itself establish trajectory accuracy.

## Validation gate before an armed claim

The first experiment must use shadow mode. It should establish that:

1. Stage 3 is invoked only on persistent LiDAR-degenerate image frames;
2. a non-zero set of real omitted visual observations passes the exact
   factorization audit;
3. weak-subspace D-efficiency and minimum-direction retention improve without
   reducing global D-efficiency; and
4. the fixed budget is respected.

Only after that gate passes should the same sequence be run with
`apply_to_estimator: true`. The armed claim then requires paired trajectory
evaluation against the identical QI baseline, plus repeated and
multi-sequence tests. If natural data provide no factor-consistent weak-space
complement, the correct result is that this recovery route is unavailable on
that frame; the implementation does not fabricate a substitute constraint.
