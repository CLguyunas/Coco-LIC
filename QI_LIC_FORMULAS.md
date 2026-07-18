# QI-LIC implementation formulas

## LiDAR quality

When the robust residual scale is ready:

`q_L = q_L_min + (q_L_max - q_L_min) * exp(-|r_pre| / sigma_L)`

Otherwise `q_L = 1`. The final residual weight is `w_L_final = w_L_base * sqrt(q_L)`, where `w_L_base` preserves the original Coco-LIC `lidar_weight` and optional correspondence scale.

## Visual quality

A frame quality is formed from the normalized PnP inlier count, PnP inlier ratio, and fundamental-matrix inlier ratio. A conservative point term uses the pre-optimization reprojection residual when its robust scale is ready. The point contribution is bounded by `qi_visual_point_quality_weight` before mapping into the visual quality interval.

## Information contribution

For both modalities, selection uses the same final residual weight as the nonlinear optimizer:

`I_j = w_j_final^2 * J_j^T * J_j`.

LiDAR and visual observations are selected separately with D-optimal greedy marginal gain. Selection stops at the hard budget, the target information coverage, or the minimum marginal gain.

## Prior consistency

The selected observations and their per-observation final weights are reused by both current-window optimization and marginalization-prior construction.
