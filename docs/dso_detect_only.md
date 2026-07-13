# DSO-DetectOnly

`DSO-DetectOnly` is the first, read-only stage of the degeneracy work. It
measures the directional information supplied by the current LiDAR
correspondences. It does **not** change:

- Ceres residual blocks or weights;
- LiDAR, IMU, or camera fusion;
- non-uniform B-spline control points;
- keyframes or the local map;
- marginalization factors or priors.

## Enable

Add the following top-level block to an odometry YAML file. The R3LIVE config
already contains this block and enables it by default on this branch.

```yaml
dso_detect_only:
    enabled: true
    output_csv: true
    use_correspondence_scale: true
    min_correspondences: 30
    analyze_every_n_scans: 1
    print_every_n_scans: 20
    relative_eigenvalue_threshold: 1.0e-3
    enter_relative_eigenvalue_threshold: 3.0e-3
    exit_relative_eigenvalue_threshold: 6.0e-3
    enter_consecutive_scans: 10
    exit_consecutive_scans: 10
    min_characteristic_range: 1.0
    max_characteristic_range: 100.0
```

When the block is absent or `enabled` is `false`, no analysis is performed.

## Output

For a bag named `degenerate_seq_02.bag`, the detector writes:

```text
config/data/degenerate_seq_02_dso_observability.csv
```

Each row contains the six eigenvalues in ascending order, their values relative
to the largest eigenvalue, the six eigenvectors, correspondence counts, scene
range normalization, condition number, raw weak-direction counts, and the
temporal degeneracy decision fields.

The temporal fields are:

- `degeneracy_score`: `-log10(relative_lambda_0)`, so larger is weaker;
- `candidate_weak_direction_num`: directions below the enter threshold;
- `degenerate_state`: hysteresis state after the current analyzed scan;
- `enter_counter` and `exit_counter`: consecutive-scan evidence accumulated
  for the next state transition.

`relative_eigenvalue_threshold` remains a legacy hard diagnostic threshold. It
only produces `weak_direction_num`; the temporal state instead enters when the
smallest relative eigenvalue remains below
`enter_relative_eigenvalue_threshold` for `enter_consecutive_scans`, and exits
when it remains above `exit_relative_eigenvalue_threshold` for
`exit_consecutive_scans`. Values between the thresholds preserve the current
state. Persistence is counted in analyzed scans, so increasing
`analyze_every_n_scans` increases the wall-clock delay.

The state order is expressed in the map frame:

```text
[rx_scaled, ry_scaled, rz_scaled, tx, ty, tz]
```

Rotation columns are divided by the median LiDAR correspondence range before
eigendecomposition. This prevents a direct, dimensionally invalid comparison
between radians and metres.

## Baseline non-interference check

Run the same bag twice, changing only `enabled`:

1. `enabled: false`
2. `enabled: true`

The detector never writes estimator state, but R3LIVE contains asynchronous
vision, map, and tree-maintenance threads. Extra diagnostic work can therefore
change scheduling, and even repeated `enabled: false` runs need not produce
byte-identical trajectories. Compare repeated ON and OFF runs instead: report
ATE/RPE mean and standard deviation, and verify that the ON result lies within
the OFF run-to-run envelope. Runtime may be slightly higher because the
detector evaluates read-only pose Jacobians.

The default enter/exit values are experimental starting points obtained from a
tunnel sequence. Keep the continuous score in all reports and revalidate the
binary thresholds before claiming cross-scene or cross-sensor generality.
