# QI-LIC regression matrix

Run all groups with the same bag start/duration, sensor calibration, knot settings, solver iterations, and evaluation alignment.

| Group | Quality | LiDAR quality | Visual quality | Selection |
|---|---:|---:|---:|---:|
| Baseline | off | on | on | off |
| LiDAR quality only | on | on | off | off |
| Visual quality only | on | off | on | off |
| Selection only | off | on | on | on |
| Full QI-LIC | on | on | on | on |

Record ATE RMSE, RPE translation/rotation, start-to-end drift, average solver time, candidate/selected counts, information coverage, log-det, and condition number.
