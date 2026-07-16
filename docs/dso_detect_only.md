# DSO-DetectOnly: dual-space diagnostics

`DSO-DetectOnly` is the read-only measurement stage of the degeneracy work.
It records two different sources of weakness:

1. **environment geometry space**: the 6DoF directional information supplied
   by the current LiDAR correspondences;
2. **non-uniform B-spline support space**: whether the correspondence
   timestamps sufficiently excite the active trajectory control-point modes.

The two spaces are kept separate so that a tunnel-like scene is not
automatically confused with poor temporal/control-point support. The analyzer,
injector, CASR router, and scheduler do **not** change:

- Ceres residual blocks or weights;
- LiDAR, IMU, or camera fusion;
- non-uniform B-spline control points;
- keyframes or the local map;
- marginalization factors or priors.

Stage 5 adds a separate CASR intervention component. It is disabled for
estimator writes unless both `casr_intervention.enabled` and
`casr_intervention.apply_to_estimator` are true. Keeping `enabled: true` and
`apply_to_estimator: false` produces a dry-run audit without changing Ceres.

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
    casr_shadow:
        enabled: true
        shadow_only: true
        output_csv: true
        environment_relative_threshold: 6.0e-3
        support_basis_relative_singular_threshold: 1.0e-6
        lift_regularization: 1.0e-6
        principal_cosine_threshold: 7.0e-1
        route_consecutive_scans: 3
        projector_consecutive_scans: 3
        projector_similarity_threshold: 8.0e-1
        activation_scheduler:
            enabled: true
            environment_full_confidence_threshold: 3.0e-3
            environment_zero_confidence_threshold: 6.0e-3
            support_full_confidence_threshold: 2.0e-2
            support_zero_confidence_threshold: 5.0e-2
            projector_full_confidence: 9.5e-1
            principal_full_confidence: 9.0e-1
            persistence_full_scans: 5
            enter_confidence: 2.5e-1
            exit_confidence: 1.0e-1
            rise_time_s: 5.0e-1
            fall_time_s: 2.0e-1
            max_dt_s: 5.0e-1
    casr_intervention:
        enabled: true
        apply_to_estimator: false
        output_csv: true
        curvature_matching_enabled: true
        curvature_target_relative_to_max: 6.0e-3
        curvature_max_added_relative_to_max: 2.0e-2
        curvature_gain: 1.0
        curvature_min_reference: 1.0e-9
        counterfactual_validation: true
        counterfactual_ratio_denominator_floor: 1.0e-6
        # Legacy fixed-weight mode only:
        base_information_weight: 1.0
        max_effective_information_weight: 1.0e+2
        min_activation_strength: 5.0e-2
        max_activation_strength: 1.0
        max_control_points: 32
        max_recovery_rank: 32
        max_basis_orthogonality_error: 1.0e-6
```

When the block is absent or `enabled` is `false`, no analysis is performed.
Setting only `support_enabled: false` preserves the environment-space detector.
The controlled injector is off by default and additionally requires
`diagnostics_only: true`; the implementation refuses any other setting.
The CASR evaluator remains restricted to `shadow_only: true`: it emits a
candidate basis and scheduler strength but has no estimator write path. The
separate Stage-5 intervention revalidates that output before it may add one
ephemeral factor.

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

## Stage 3: CASR-v2 knot-space shadow recovery

CASR-v1 folded every weak `6K` spline mode into a 6-by-6 matrix by summing
per-knot outer products. Controlled compression showed that this can become an
almost isotropic, full-rank pose space: temporal signs and control-point
localization disappear, and every environment direction then appears to be a
trivial common direction. CASR-v2 removes that fold. Both causes are compared
in the active control-point coordinates

```text
z = [r*dtheta_0, dp_0, ..., r*dtheta_(K-1), dp_(K-1)],
```

where `r` is the same characteristic range used by the environment detector.
The generalized spline modes are transformed to `z` and Euclidean-
orthonormalized to obtain the support weak basis `U_s^K`. The support-quality
threshold still decides which physical modes are weak;
`support_basis_relative_singular_threshold` only removes numerical linear
dependence.

The environment weak basis `U_e` remains the 6DoF map-frame eigenvectors whose
relative eigenvalues are inside `environment_relative_threshold`. For every
uniform reference timestamp `t_j`, CASR constructs the exact mapping `B_j`
from scaled NURBS knot perturbations to the scaled map-frame LiDAR pose. Its
rotation block converts the NURBS right perturbation to the detector's
map-frame left perturbation, and its translation block includes the LiDAR-IMU
lever arm. Each environment direction is lifted over the entire scan by

```text
x_e = (mean_j B_j^T B_j + lambda I)^(-1) mean_j B_j^T e,
```

with `lambda` controlled by `lift_regularization` relative to the largest
reference-information eigenvalue. Orthonormalizing the lifted columns gives
`U_e^K`. `environment_lift_residual` records how well one knot perturbation
reproduces the same map-frame direction over the scan.

CASR-v2 computes the singular values of

```text
(U_e^K)^T U_s^K.
```

These are the control-point-space principal-angle cosines. The normalized
squared sum is `overlap_score`. A common direction is retained only when its
cosine is at least `principal_cosine_threshold`. Routing is therefore:

| Cause | Raw shadow route | Knot-space candidate |
|---:|---|---|
| `0` | `inactive` | empty |
| `1` | `environment_candidate` | lifted environment weak basis |
| `2` | `support_candidate` | generalized support weak basis |
| `3`, common direction exists | `coupled_common_candidate` | principal common basis |
| `3`, no reliable common direction | `coupled_conflict` | empty; defer action |

The explicit conflict route remains essential: a coupled label alone cannot
justify inventing a recovery direction. `recovery_rank` now means knot-space
rank. The logged 6-by-6 recovery matrix is only the candidate mapped at the
scan's representative timestamp; it is a compact inspection view and is not
used for routing.

CASR-v2 also adds an independent temporal gate. `stable_route` changes only
after `route_consecutive_scans` identical raw candidates. Candidate continuity
is measured after aligning overlapping **global knot indices**. Each basis is
first restricted to the common control points and re-orthonormalized there.
This removes artificial energy loss caused only by a rolling spline window
dropping one boundary knot:

```text
U_previous_overlap = orth(restrict(U_previous, common_knots))
U_current_overlap  = orth(restrict(U_current, common_knots))
affinity = ||U_previous_overlap^T U_current_overlap||_F^2
forced_overlap = max(0, rank_previous_overlap + rank_current_overlap
                        - overlap_ambient_dimension)
similarity = (affinity - forced_overlap) /
             (max(rank_previous_overlap, rank_current_overlap)
                - forced_overlap).
```

The result is clamped to `[0, 1]`. Here the overlap ambient dimension is six
times the number of common control points. Two high-rank subspaces in this
ambient space must share at least `forced_overlap` dimensions even when their
remaining directions are unrelated. Subtracting this unavoidable Grassmann
intersection prevents rank alone from satisfying the temporal gate, while the
max-rank denominator still penalizes a genuine rank change.

`temporal_overlap_control_point_num`, both restricted ranks,
`temporal_forced_overlap_rank`, and the unnormalized
`temporal_projector_affinity` make the decision independently reproducible.
The overlap size is zero for the first candidate after a route change because
no previous basis exists. CSV `method_version` is
`knot_space_v2_overlap_debiased_scheduler_v1` after the Stage-4 scheduler is
enabled; the overlap definition itself is unchanged.

`recovery_ready` becomes true only after the stable route matches the raw route
and the similarity remains above `projector_similarity_threshold` for
`projector_consecutive_scans`. Raw routes remain logged, so the filter cannot
hide a flickering detector. The real and injected-copy paths keep separate
temporal states.

### Cause-aware pre-intervention scheduler

The Stage-4 scheduler remains inside the shadow evaluator. It does not add a
factor or alter a residual. Its purpose is to turn a binary `recovery_ready`
event into an auditable continuous candidate strength. Stage 5 may consume
only the real-data scheduler result after repeating every safety check.

Four confidence terms are computed in `[0, 1]`:

1. environment confidence decreases linearly from one at the detector enter
   threshold to zero at the exit threshold;
2. support confidence uses the same descending interpolation between the
   support enter and exit quality thresholds;
3. temporal confidence increases from zero at
   `projector_similarity_threshold` to one at
   `projector_full_confidence`;
4. persistence confidence increases from the first `recovery_ready` scan to
   one at `persistence_full_scans`.

For `environment_candidate` and `support_candidate`, cause confidence is the
matching detector confidence. A `coupled_common_candidate` additionally uses
the minimum environment/support confidence and a principal-angle confidence
that rises from `principal_cosine_threshold` to
`principal_full_confidence`. The raw scheduler confidence is conservative:

```text
raw_confidence = min(cause_confidence,
                     temporal_confidence,
                     persistence_confidence).
```

An enter/exit confidence pair provides a second hysteresis layer. While the
candidate remains valid, target strength follows the raw confidence and the
reported activation strength approaches it with time-based rise/fall slew
limits. Using elapsed seconds rather than scans makes the slew independent of
LiDAR frequency. Invalid input, `inactive`, `coupled_conflict`, route mismatch,
missing recovery basis, or `recovery_ready == false` immediately blocks and
resets the scheduler. Real and injected-copy schedulers reuse their already
independent temporal states.

When enabled, CASR writes:

```text
config/data/degenerate_seq_02_casr_shadow.csv
```

Each row contains real and optional injected-copy blocks. In addition to the
raw cause/route, ranks, principal cosines, overlap, exclusive ratios and the
representative 6DoF matrix, the v2 block records `method_version`, active knot
start/dimension, lift residual, basis orthogonality error, stable route,
pending-route count, temporal similarity, temporal overlap size, restricted
ranks, forced-overlap rank, raw projector affinity, consistency count, and
`recovery_ready`. The scheduler extension then records its state code/name,
eligibility and active flags, all confidence components, raw confidence,
target strength, slewed activation strength, and elapsed time. The original
81-column observability CSV and 45-column injection CSV remain unchanged.

`CASR-Shadow` still does not modify Ceres, the spline, measurements, the map,
or the marginalization prior. A configuration with `shadow_only: false` is
refused. Its output is consumed only by the separately armed intervention
described next.

## Stage 5: CASR estimator intervention

The intervention implements an anisotropic soft anchor in the exact `6K`
knot space. Immediately after IMU propagation and before the LIC iterations,
it copies only the recent control-point suffix. If the final-iteration CASR
result is valid, schedulable, route-consistent, `recovery_ready`, active, and
numerically well formed, the factor uses the pre-LIC control points as its
reference. For each covered control point it forms

```text
delta = [r*Log(R_reference^-1 R_current),
         p_current - p_reference].
```

Before adding a factor, v2 asks Ceres for the robustified Jacobian of the
unmodified final-LIC problem with respect to the active CASR knots. Rotation
columns are converted from radians to the same characteristic-range metric as
`[r*dtheta, dp]`. Let `H_s` be this scaled conditional knot Hessian. The code
diagonalizes the recovery-space curvature

```text
C_r = B_r^T H_s B_r = U diag(h_i) U^T
B_w = B_r U
target = curvature_target_relative_to_max * lambda_max(H_s)
w_i = clamp(activation * curvature_gain * (target - h_i),
            0,
            curvature_max_added_relative_to_max * lambda_max(H_s))
residual_i = sqrt(w_i) * B_w(:,i)^T delta.
```

Thus every recovery direction receives only its measured curvature deficit;
directions already above the target receive zero added information. If all
directions are sufficient the frame is logged as `curvature_sufficient` and no
factor is added. Invalid or near-zero reference curvature fails closed as
`curvature_invalid`. The legacy scalar `base_information_weight` path remains
available only when `curvature_matching_enabled: false`.

Only diagnosed recovery coordinates are damped. Updates in the orthogonal,
observable complement remain unconstrained by this factor. The SO(3) Jacobian
uses the exact right-Jacobian inverse corresponding to Coco-LIC's right
perturbation. The factor is added only to the final LIC refinement and is never
inserted into `UpdateLICPrior`; a transient diagnosis cannot pollute the
long-lived marginalization prior.

The intervention repeats the following hard gates independently of the
scheduler: valid CASR input, one of the three schedulable routes, raw/stable
route agreement, `recovery_ready`, active scheduler, activation floor, finite
orthonormal basis, configured rank/dimension bounds, current control-point
range, an explicit `RealMeasurements` provenance tag, and an exact
timestamp-matched pre-LIC reference. `inactive`, `coupled_conflict`, stale
routes, diagnostics-copy results, or missing references always produce zero
estimator factors. The provenance check is repeated at the estimator boundary;
it does not rely only on the caller selecting the real result.

With `counterfactual_validation: true`, every parameter block registered in
the final Ceres problem is snapshotted. The unmodified baseline is solved and
measured first; then the exact pre-solve snapshot is restored and the CASR
problem is solved. This produces a same-frame, same-initial-state comparison.
The baseline solution snapshot is also retained: if the CASR solve is unusable,
the factor is removed and the already validated baseline solution is restored.
If a baseline snapshot is unavailable, the previous remove-and-resolve fallback
is used. Thus a failed CASR attempt is never committed silently.

With `casr_intervention.enabled: true`, a fourth audit file is written:

```text
data/degenerate_seq_02_casr_intervention.csv
```

Its `state` distinguishes safety blocks, `dry_run`, `applied`, recovered
solver failure, and unrecovered solver failure. It records source provenance,
requested/used activation, effective information, control-point range/rank,
factor-added/committed flags, primary/fallback solver status, projected LIC
curvatures, per-direction added-information summaries, and the total,
projected, and orthogonal scaled increments. The decisive effectiveness fields
are `projected_casr_over_counterfactual` and
`orthogonal_casr_over_counterfactual`, which compare the CASR and unmodified
solutions of the same frame from the same initial state.

Use the two-switch sequence deliberately:

1. `enabled: true`, `apply_to_estimator: false`: validate curvature extraction,
   routing, and the automatically computed direction weights without changing
   the estimator;
2. `enabled: true`, `apply_to_estimator: true`: enable same-frame
   counterfactual solving and the curvature-matched factor;
3. use the logged same-frame ratio as the primary factor-effect test, then use
   complete-trajectory ATE/RPE as the end-to-end accuracy test. Do not tune the
   detector thresholds during this validation.

Audit one run, or align a dry-run and armed run by scan timestamp, with:

```shell
python3 tools/analyze_casr_intervention.py \
  --run dry=data/sequence_dry_casr_intervention.csv \
  --run armed=data/sequence_armed_casr_intervention.csv
```

The command exits nonzero for source leakage, inconsistent factor/commit
flags, a factor added while `apply_to_estimator` is false, non-finite metrics,
or invalid fallback-state combinations. It reports dry-run post/pre metrics,
curvature-matched added information, and median/p90 CASR-to-same-frame-baseline
ratios for the diagnosed projection and its orthogonal complement.

## Baseline non-interference check

For detector/shadow non-interference, keep
`casr_intervention.apply_to_estimator: false` and run the same bag twice,
changing only detector `enabled`:

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

Together, severity `0.5` and `0.75` runs provide the cause cases needed to
validate the CASR shadow router. For the first CASR replay, keep the validated
`timestamp_compression`, severity `0.75`, phase `0.0-1.0`, and seed `42`, then
check that support-only frames select `support_candidate`, coupled frames
select `coupled_common_candidate` or the explicit conflict route, every
knot basis is orthonormal within numerical tolerance, environment-lift
residuals remain finite, the stable route suppresses isolated raw-route
flips, consecutive candidates report a nonzero temporal overlap, stable
environment segments can reach `recovery_ready`, and `recovery_ready` is never
asserted during an unstable transition. For every compared pair verify
`forced_overlap == max(0, previous_rank + current_rank - 6 * overlap_knots)`
and recompute the logged similarity from the raw affinity. In particular, two
rank-12 subspaces in an 18-dimensional three-knot overlap have a forced rank
of six; those six directions alone must yield similarity zero, not 0.5.
For scheduler validation, verify that every unsafe/conflict/not-ready row has
zero activation, confidence values remain in `[0, 1]`, the real and injected
paths evolve independently, rise/fall changes respect their elapsed-time slew
limits, and a threshold-edge ready event produces either
`below_enter_confidence` or only negligible target strength. Stable,
high-confidence environment segments should ramp smoothly instead of jumping
from zero to full strength.
Use at least five repeated detector-OFF and shadow-ON runs to report ATE/RPE
mean and standard deviation; two extrema alone are not a non-interference
test. This validates only the shadow and dry-run paths. Armed intervention
must be evaluated separately with repeated OFF/dry-run/armed trajectories,
cause-specific ATE/RPE, projected-increment suppression, solver failures,
runtime, and non-degenerate-sequence regression checks.
