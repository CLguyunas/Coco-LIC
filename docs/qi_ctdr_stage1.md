# QI-CTDR Stage 1: continuous-time LiDAR observability shadow audit

This branch starts from `master`. Stage 1 intentionally implements only the
read-only LiDAR diagnostic needed by the agreed QI-CTDR plan. It does not add a
prior, add a residual, change a weight, select an observation, or write a weak
direction back to Ceres.

## What is measured

For the complete LiDAR correspondence pool produced before the final LIC solve,
the analyzer evaluates the existing scalar `LoamFeatureFactorNURBS` Jacobian at
each point timestamp. The local tangent columns of every free non-uniform spline
control point are assembled into the LiDAR-only control-point information
matrix.

The control-point covariance is propagated to a reference pose on the
continuous trajectory:

```text
Sigma_ref = A_ref * inverse(H_lidar + numeric_floor * I) * A_ref^T
```

`A_ref` is the exact non-uniform NURBS pose-to-control-point Jacobian. Rotation
and translation are made comparable with a characteristic length derived from
the median LiDAR point range. The eigenvalue ratios of
`inverse(Sigma_ref)` are therefore a LiDAR-only localizability diagnostic at the
reference trajectory time, not a hardware-failure detector and not a
"spline-support degradation" detector.

The detector uses all valid LiDAR candidates, not an information-selected
subset. This prevents a future observation manager from creating its own
apparent degeneracy.

## Enable the shadow audit

```yaml
ct_degeneracy:
  enabled: true
  apply: false
  output_csv: true
  weak_eigenvalue_ratio: 3.0e-3
```

`lidar_iter` must be at least 2. The first LIC solve supplies a warm start; the
diagnostic is evaluated after re-association and immediately before the final
LIC solve.

The output is:

```text
data/<bag-name>_ct_lidar_observability.csv
```

Audit it with:

```bash
python3 tools/analyze_ct_lidar_observability.py \
  --run tunnel=data/Tunneling_tunnel4_gamma_ct_lidar_observability.csv
```

## Stage gate

Estimator intervention must remain disabled until all of the following hold:

1. analytic LiDAR Jacobians agree with local-manifold finite differences;
2. weak/strong eigen-directions agree with symmetric cost perturbations;
3. natural degenerate sequences show persistent weak directions while
   non-degenerate sequences do not produce comparable false activation;
4. a later fixed-budget visual repair improves weak-direction information
   relative to the QI-only set without reducing its global information
   coverage.

