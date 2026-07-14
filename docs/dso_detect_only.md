# DSO-DetectOnly: dual-space diagnostics

`DSO-DetectOnly` is the read-only measurement stage of the degeneracy work.
It records two different sources of weakness:

1. **environment geometry space**: the 6DoF directional information supplied
   by the current LiDAR correspondences;
2. **non-uniform B-spline support space**: whether the correspondence
   timestamps sufficiently excite the active trajectory control-point modes.

The two spaces are kept separate so that a tunnel-like scene is not
automatically confused with poor temporal/control-point support. The analyzer
does **not** change:

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
    support_enabled: true
    support_reference_samples_per_interval: 32
    support_max_control_points: 32
    support_enter_quality_threshold: 2.0e-2
    support_exit_quality_threshold: 5.0e-2
    support_enter_consecutive_scans: 10
    support_exit_consecutive_scans: 10
    support_injection:
        enabled: false
        diagnostics_only: true
        output_csv: true
        mode: timestamp_compression
        severity: 0.5
        phase_start: 0.0
        phase_end: 1.0
        random_seed: 42
```

When the block is absent or `enabled` is `false`, no analysis is performed.
Setting only `support_enabled: false` preserves the environment-space detector.
The controlled injector is off by default and additionally requires
`diagnostics_only: true`; the implementation refuses any other setting.

## Output

For a bag named `degenerate_seq_02.bag`, the detector writes:

```text
config/data/degenerate_seq_02_dso_observability.csv
```

Each row contains the six environment-space eigenvalues in ascending order,
their values relative to the largest eigenvalue, the six eigenvectors,
correspondence counts, scene range normalization, condition number, raw
weak-direction counts, temporal decisions, and the spline-support summary.

The environment temporal fields are:

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

The environment state order is expressed in the map frame:

```text
[rx_scaled, ry_scaled, rz_scaled, tx, ty, tz]
```

Rotation columns are divided by the median LiDAR correspondence range before
eigendecomposition. This prevents a direct, dimensionally invalid comparison
between radians and metres.

## Exact spline-support space

For every accepted correspondence at time `t`, the analyzer uses the same
four rotation knots, four position knots, non-uniform blending matrices, and
analytic SO(3) knot Jacobians as `LoamFeatureFactorNURBS`. Let `B(t)` be the
6-by-`6K` mapping from the active `K` knot perturbations to the interpolated
trajectory pose perturbation. The observed support matrix is

```text
H_observed = mean_t B(t)^T B(t).
```

This deliberately removes the point-to-plane/line geometry Jacobian: geometry
is already measured by the 6DoF matrix above. Correspondence scale is retained
as timestamp evidence weight.

Raw B-spline Gram matrices are intrinsically ill-conditioned and their size
changes when a scan crosses more knot intervals. Therefore the detector also
builds `H_reference` by uniformly sampling every non-uniform knot interval
spanned by the scan, using the current knots and exactly the same analytic
mapping. It eigendecomposes the whitened matrix

```text
Q = H_reference^(-1/2) H_observed H_reference^(-1/2).
```

`support_quality_min` is the smallest eigenvalue of `Q`. A value near `1`
means the effective LiDAR timestamps support the active control modes similarly
to ideal uniform temporal sampling; a value near `0` identifies a control-point
combination that is poorly supported. This is a relative support diagnostic,
not another absolute Hessian threshold.

Important spline fields are:

- `support_control_point_num`, `support_interval_num`, and
  `support_dimension`: active support size (`support_dimension = 6K`);
- `support_effective_rank` and `support_weak_direction_num`: modes above/below
  `support_enter_quality_threshold`;
- `support_time_span_s`, `support_min_knot_dt_s`, and
  `support_max_knot_dt_s`: temporal context;
- `support_quality_min`, `support_condition_number`, and `support_score`;
- `support_degenerate_state`, `support_enter_counter`, and
  `support_exit_counter`: an independent persistence/hysteresis state;
- `support_weakest_knot_index`, `support_weakest_knot_energy_ratio`,
  `support_weakest_rotation_ratio`, and `support_boundary_energy_ratio`:
  localization and composition of the weakest generalized mode.

`support_score` is `-log10(min(support_quality_min, 1))`. The default support
enter/exit thresholds (`0.02/0.05`) are conservative starting points, not
published universal values.

## Dual-space cause label

`degeneracy_cause` combines the two independent hysteresis states only when
their inputs are valid:

| Value | Label | Interpretation |
|---:|---|---|
| `-1` | invalid | one of the enabled diagnostics has insufficient evidence |
| `0` | healthy | neither space is persistently weak |
| `1` | environment | LiDAR geometry is weak, spline support is healthy |
| `2` | spline support | geometry is healthy, timestamp/control support is weak |
| `3` | coupled | both spaces are persistently weak |

The label is logging-only. It does not select residual weights or modify the
trajectory.

## Controlled support-degradation injection

The optional `support_injection` block is an experimental validation harness,
not part of the online detector or CASR recovery path. It copies only the
accepted `(timestamp, correspondence_weight)` pairs and transforms that copy
before a second spline-support analysis. The original correspondences and
timestamps still feed the environment diagnostic, the real spline-support
diagnostic, and the estimator without modification.

When enabled, a second file is written next to the main CSV:

```text
config/data/degenerate_seq_02_dso_support_injection.csv
```

The main `*_dso_observability.csv` keeps its existing 81 columns and its real
`degeneracy_cause`. The injection CSV records both original and injected
support metrics, sample counts, retained/time-span ratios, injected hysteresis,
the number of preserved boundary anchors, and `injected_degeneracy_cause`.
The injected cause combines the **real**
environment state with the **injected-copy** support state:

| Real environment | Injected support | Injected cause |
|---|---|---:|
| healthy | healthy | `0` |
| weak | healthy | `1` |
| healthy | weak | `2` |
| weak | weak | `3` |

All phase parameters are normalized independently inside each scan:

```text
s_i = (t_i - t_min) / (t_max - t_min).
```

This avoids hard-coding a bag timestamp, scan duration, LiDAR rate, or the
`Tunneling_tunnel4_gamma` sequence. Available modes are:

- `timestamp_compression`: preserve all earliest/latest timestamp samples as
  boundary anchors, then move the remaining timestamps in
  `[phase_start, phase_end]` toward the phase-window center; `severity=1`
  collapses the selected non-anchor samples to the center;
- `phase_dropout`: remove selected-window samples with probability `severity`;
- `boundary_dropout`: remove samples outside the selected central window with
  probability `severity`;
- `temporal_thinning`: remove samples across the whole scan with probability
  `severity`, mainly testing evidence-count/invalid handling.

Dropout uses a deterministic hash of the timestamp, sample index, and
`random_seed`, so the same input and configuration produce the same diagnostic
copy. No global random generator or estimator scheduling state is touched.

The boundary anchors keep the original timestamp span and active
control-point range fixed during compression. This prevents a compressed scan
from accidentally landing on a smaller set of complete knot intervals and
appearing healthy only because its reference dimension also shrank. For
`timestamp_compression`, `timestamp_span_ratio` should therefore remain `1`,
and `boundary_anchor_sample_num` should normally be at least `2`.

The implementation is sequence-independent, but empirical generalization must
still be demonstrated. Use fixed phase parameters and seeds on multiple bags,
sweep severity, and verify that injected support quality responds monotonically
while the original support fields and trajectory accuracy stay inside repeated
baseline variation.

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
detector evaluates read-only pose and control-point Jacobians.

The environment enter/exit values were validated on one tunnel sequence. The
spline-support thresholds still require a fresh run and data-driven
calibration. Keep both continuous scores in all reports and revalidate binary
thresholds before claiming cross-scene or cross-sensor generality.

## Stage-2 validation run

Run `GEODE/Tunneling_tunnel4_gamma` once with both diagnostics enabled and
provide the new `*_dso_observability.csv`. First verify that
`support_valid == 1` for ordinary scans and inspect the distributions of
`support_quality_min`, support rank loss, boundary-mode energy, and all four
cause classes. Do not tune thresholds from trajectory ATE alone: tune them from
stable temporal segments and confirm that the continuous support signal is not
merely duplicating `relative_lambda_0`.

## Injection validation and CASR routing table

Start with `timestamp_compression`, fixed `phase_start/end=0.0/1.0`, seed 42,
and severities `0.0, 0.25, 0.5, 0.75, 1.0`. For every run preserve both CSVs
and the trajectory. A valid harness should show:

1. severity 0 reproduces the original support metrics up to numerical error;
2. anchored compression keeps `timestamp_span_ratio == 1`, reports at least
   two boundary anchors, and preserves the original active control-point range;
3. increasing compression reduces `injected_support_quality_min` in affected
   scans without changing `original_support_quality_min`;
4. the real environment fields are identical in definition and see no injected
   timestamps;
5. `enabled: false` creates no injection CSV and preserves the previous path;
6. ON/OFF trajectory differences remain within repeated-run scheduling noise.

Together, an original run and an injected run provide the four cause cases
needed to test the future cause-adaptive CASR router. They do not yet validate
CASR recovery performance: CASR remains a later estimator intervention and
must be evaluated separately against cause-specific accuracy and consistency
metrics.
