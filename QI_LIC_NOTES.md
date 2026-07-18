# QI-LIC implementation notes

This branch refactors the previous QIM observation-management prototype into QI-LIC (Quality-aware and Information-complementary LIC).

## Implemented

- Removed LiDAR/visual cross-window memory from the executable method path.
- Added bounded LiDAR non-attenuating quality mapping with residual-scale warm-up.
- Added visual frame-level and conservative point-level quality calibration.
- Constructed selection information matrices using the same final residual weights used by optimization.
- Added modality-separated D-optimal greedy selection with hard budgets, information-coverage stopping, and minimum-gain stopping.
- Reused the selected observations and their per-observation weights in both current optimization and marginalization prior construction.
- Kept the original Coco-LIC path when both quality and selection are disabled.

## Validation status

The payload application workflow completed its static consistency checks. A full ROS/Catkin build and dataset regression run are still required in the target Coco-LIC environment.
