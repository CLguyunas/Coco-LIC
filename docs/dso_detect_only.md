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
    min_characteristic_range: 1.0
    max_characteristic_range: 100.0
```

When the block is absent or `enabled` is `false`, no analysis is performed.

## Output

For a bag named `degenerate_seq_02.bag`, the detector writes:

```text
data/degenerate_seq_02_dso_observability.csv
```

Each row contains the six eigenvalues in ascending order, their values relative
to the largest eigenvalue, the six eigenvectors, correspondence counts, scene
range normalization, condition number, and number of weak directions.

The state order is expressed in the map frame:

```text
[rx_scaled, ry_scaled, rz_scaled, tx, ty, tz]
```

Rotation columns are divided by the median LiDAR correspondence range before
eigendecomposition. This prevents a direct, dimensionally invalid comparison
between radians and metres.

## Baseline invariance check

Run the same bag twice, changing only `enabled`:

1. `enabled: false`
2. `enabled: true`

The two TUM trajectory files should be numerically identical. Runtime may be
slightly higher when diagnostics are enabled because the detector evaluates the
read-only pose Jacobians.
