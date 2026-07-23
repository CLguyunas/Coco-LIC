#!/usr/bin/env python3
"""Audit QI observation-management CSV files.

The audit is deliberately limited to properties visible inside QI itself:
weight semantics, budget compliance, retained information, and residual
statistics. Trajectory accuracy remains a separate experiment.
"""

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path


def quantile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def summary(values):
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"count": 0, "median": None, "p10": None, "p90": None}
    return {
        "count": len(finite),
        "median": statistics.median(finite),
        "p10": quantile(finite, 0.10),
        "p90": quantile(finite, 0.90),
    }


def as_float(row, key):
    return float(row[key])


def as_int(row, key):
    return int(float(row[key]))


def safe_ratio(numerator, denominator):
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if abs(denominator) <= 1.0e-12:
        return None
    return numerator / denominator


def load_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        required = {
            "scan_time_ns",
            "optimization_success",
            "quality_enabled",
            "selection_enabled",
            "lidar_candidates",
            "lidar_selected",
            "lidar_q_min",
            "lidar_q_mean",
            "lidar_q_max",
            "lidar_weight_ratio_min",
            "lidar_weight_ratio_mean",
            "lidar_weight_ratio_max",
            "lidar_info_coverage",
            "lidar_pre_median",
            "lidar_post_median",
            "visual_candidates",
            "visual_selected",
            "visual_info_coverage",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError("missing columns: " + ", ".join(missing))
        return rows


def audit(label, path):
    rows = load_rows(path)
    if not rows:
        raise ValueError("CSV has no data rows")

    quality_modes = {as_int(row, "quality_enabled") for row in rows}
    selection_modes = {as_int(row, "selection_enabled") for row in rows}
    mode_stable = len(quality_modes) == 1 and len(selection_modes) == 1
    quality_enabled = bool(next(iter(quality_modes))) if quality_modes else False
    selection_enabled = (
        bool(next(iter(selection_modes))) if selection_modes else False
    )

    structural_errors = []
    lidar_selection_ratios = []
    visual_selection_ratios = []
    lidar_post_pre = []
    visual_post_pre = []
    lidar_coverage = []
    visual_coverage = []
    lidar_q_mean = []
    visual_q_mean = []
    lidar_weight_mean = []
    visual_weight_mean = []
    optimization_success = 0
    image_rows = 0

    for index, row in enumerate(rows, start=2):
        lidar_candidates = as_int(row, "lidar_candidates")
        lidar_selected = as_int(row, "lidar_selected")
        visual_candidates = as_int(row, "visual_candidates")
        visual_selected = as_int(row, "visual_selected")
        process_image = as_int(row, "process_image")
        success = as_int(row, "optimization_success")
        optimization_success += int(success != 0)
        image_rows += int(process_image != 0)

        if lidar_candidates < 0 or lidar_selected < 0:
            structural_errors.append(f"row {index}: negative LiDAR count")
        if visual_candidates < 0 or visual_selected < 0:
            structural_errors.append(f"row {index}: negative visual count")
        if lidar_selected > lidar_candidates:
            structural_errors.append(
                f"row {index}: LiDAR selected exceeds candidates"
            )
        if visual_selected > visual_candidates:
            structural_errors.append(
                f"row {index}: visual selected exceeds candidates"
            )
        if not selection_enabled and lidar_selected != lidar_candidates:
            structural_errors.append(
                f"row {index}: selection disabled but LiDAR set changed"
            )
        if (
            not selection_enabled
            and process_image
            and visual_selected != visual_candidates
        ):
            structural_errors.append(
                f"row {index}: selection disabled but visual set changed"
            )

        lidar_selection_ratios.append(
            lidar_selected / lidar_candidates
            if lidar_candidates > 0
            else 1.0
        )
        if process_image and visual_candidates > 0:
            visual_selection_ratios.append(
                visual_selected / visual_candidates
            )

        lidar_coverage.append(as_float(row, "lidar_info_coverage"))
        if process_image and visual_candidates > 0:
            visual_coverage.append(as_float(row, "visual_info_coverage"))
        lidar_q_mean.append(as_float(row, "lidar_q_mean"))
        lidar_weight_mean.append(as_float(row, "lidar_weight_ratio_mean"))
        if process_image and visual_candidates > 0:
            visual_q_mean.append(as_float(row, "visual_q_mean"))
            visual_weight_mean.append(
                as_float(row, "visual_weight_ratio_mean")
            )

        lidar_ratio = safe_ratio(
            as_float(row, "lidar_post_median"),
            as_float(row, "lidar_pre_median"),
        )
        if lidar_ratio is not None:
            lidar_post_pre.append(lidar_ratio)
        if process_image and visual_candidates > 0:
            visual_ratio = safe_ratio(
                as_float(row, "visual_post_median"),
                as_float(row, "visual_pre_median"),
            )
            if visual_ratio is not None:
                visual_post_pre.append(visual_ratio)

        for prefix in ("lidar", "visual"):
            q_min = as_float(row, f"{prefix}_q_min")
            q_mean = as_float(row, f"{prefix}_q_mean")
            q_max = as_float(row, f"{prefix}_q_max")
            if not (q_min > 0.0 and q_min <= q_mean <= q_max):
                structural_errors.append(
                    f"row {index}: invalid {prefix} q ordering"
                )
            coverage = as_float(row, f"{prefix}_info_coverage")
            if coverage < -1.0e-9 or coverage > 1.0 + 1.0e-9:
                structural_errors.append(
                    f"row {index}: invalid {prefix} coverage"
                )

        if not quality_enabled:
            for prefix in ("lidar", "visual"):
                if abs(as_float(row, f"{prefix}_q_mean") - 1.0) > 1.0e-9:
                    structural_errors.append(
                        f"row {index}: quality disabled but {prefix} q != 1"
                    )
                if (
                    abs(
                        as_float(row, f"{prefix}_weight_ratio_mean") - 1.0
                    )
                    > 1.0e-9
                ):
                    structural_errors.append(
                        f"row {index}: quality disabled but "
                        f"{prefix} weight ratio != 1"
                    )

    passed = mode_stable and not structural_errors
    result = {
        "rows": len(rows),
        "passed": passed,
        "mode": {
            "quality": quality_enabled,
            "selection": selection_enabled,
            "stable": mode_stable,
        },
        "optimization_success_fraction": optimization_success / len(rows),
        "image_rows": image_rows,
        "lidar_selected/candidates": summary(lidar_selection_ratios),
        "visual_selected/candidates": summary(visual_selection_ratios),
        "lidar_information_coverage": summary(lidar_coverage),
        "visual_information_coverage": summary(visual_coverage),
        "lidar_q_mean": summary(lidar_q_mean),
        "visual_q_mean": summary(visual_q_mean),
        "lidar_sqrt_information_multiplier": summary(lidar_weight_mean),
        "visual_sqrt_information_multiplier": summary(visual_weight_mean),
        "lidar_post/pre_residual": summary(lidar_post_pre),
        "visual_post/pre_residual": summary(visual_post_pre),
        "coverage_at_0.95": {
            "lidar_fraction": (
                sum(value >= 0.95 for value in lidar_coverage)
                / len(lidar_coverage)
                if lidar_coverage
                else None
            ),
            "visual_fraction": (
                sum(value >= 0.95 for value in visual_coverage)
                / len(visual_coverage)
                if visual_coverage
                else None
            ),
        },
        "structural_errors": structural_errors[:20],
    }
    print(f"[{label}] rows={result['rows']} passed={passed}")
    for key, value in result.items():
        if key in {"rows", "passed", "structural_errors"}:
            continue
        print(f"  {key}={value}")
    if structural_errors:
        print(f"  structural_errors={result['structural_errors']}")
    return passed


def parse_run(argument):
    if "=" not in argument:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, raw_path = argument.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label, Path(raw_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run,
        metavar="LABEL=PATH",
    )
    args = parser.parse_args()

    all_passed = True
    for label, path in args.run:
        try:
            all_passed = audit(label, path) and all_passed
        except (OSError, ValueError, KeyError) as error:
            print(f"QI audit input error [{label}]: {error}", file=sys.stderr)
            all_passed = False
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
