#!/usr/bin/env python3
"""Evaluate a TUM trajectory without requiring evo.

The estimator timestamps are evaluated only inside the ground-truth time
range. Ground-truth positions are linearly interpolated at estimator
timestamps, then an SE(3) (default), Sim(3), or no alignment is applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def describe(values: np.ndarray) -> Dict[str, Optional[float]]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "rmse": None,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
        "max": float(np.max(values)),
    }


def read_tum(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = []
    positions = []
    quaternions = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.replace(",", " ").split()
            if len(fields) < 4:
                # Coco-LIC can leave one timestamp-only final line when the
                # bag ends between buffered trajectory samples.
                continue
            values = [float(value) for value in fields]
            if not all(math.isfinite(value) for value in values[:4]):
                continue
            times.append(values[0])
            positions.append(values[1:4])
            if len(values) >= 8:
                quaternions.append(values[4:8])
            else:
                quaternions.append([math.nan] * 4)
    if len(times) < 3:
        raise ValueError(f"{path}: fewer than three valid trajectory rows")
    order = np.argsort(np.asarray(times))
    return (
        np.asarray(times, dtype=float)[order],
        np.asarray(positions, dtype=float)[order],
        np.asarray(quaternions, dtype=float)[order],
    )


def interpolate_positions(
    source_times: np.ndarray,
    source_positions: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(query_times, source_times, source_positions[:, axis])
            for axis in range(3)
        ]
    )


def umeyama(
    source: np.ndarray, target: np.ndarray, with_scale: bool
) -> Tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.shape[1] != 3:
        raise ValueError("alignment inputs must be matching Nx3 arrays")
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (
        target_centered.T @ source_centered / float(source.shape[0])
    )
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u_matrix) * np.linalg.det(vt_matrix) < 0.0:
        sign[-1] = -1.0
    rotation = u_matrix @ np.diag(sign) @ vt_matrix
    scale = 1.0
    if with_scale:
        variance = np.mean(np.sum(source_centered * source_centered, axis=1))
        if variance <= 1.0e-15:
            raise ValueError("estimated trajectory has zero spatial variance")
        scale = float(np.dot(singular_values, sign) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def evaluate(
    estimate_path: Path,
    ground_truth_path: Path,
    alignment: str,
    time_offset: float,
    rpe_delta: float,
):
    estimate_times, estimate_positions, _ = read_tum(estimate_path)
    gt_times, gt_positions, _ = read_tum(ground_truth_path)
    shifted_times = estimate_times + time_offset
    overlap = (shifted_times >= gt_times[0]) & (
        shifted_times <= gt_times[-1]
    )
    shifted_times = shifted_times[overlap]
    estimate_positions = estimate_positions[overlap]
    if shifted_times.size < 3:
        raise ValueError("fewer than three estimate samples overlap GT")
    gt_interpolated = interpolate_positions(
        gt_times, gt_positions, shifted_times
    )

    if alignment == "none":
        scale = 1.0
        rotation = np.eye(3)
        translation = np.zeros(3)
    else:
        scale, rotation, translation = umeyama(
            estimate_positions,
            gt_interpolated,
            with_scale=alignment == "sim3",
        )
    aligned = (
        scale * (rotation @ estimate_positions.T).T + translation
    )
    ate = np.linalg.norm(aligned - gt_interpolated, axis=1)

    rpe = np.full(shifted_times.shape, np.nan, dtype=float)
    if rpe_delta > 0.0:
        for first in range(shifted_times.size):
            target_time = shifted_times[first] + rpe_delta
            second = int(np.searchsorted(shifted_times, target_time))
            if second >= shifted_times.size:
                break
            if second == first:
                continue
            estimate_increment = aligned[second] - aligned[first]
            gt_increment = gt_interpolated[second] - gt_interpolated[first]
            rpe[first] = float(
                np.linalg.norm(estimate_increment - gt_increment)
            )

    path_length = float(
        np.sum(np.linalg.norm(np.diff(gt_interpolated, axis=0), axis=1))
    )
    final_drift = float(np.linalg.norm(aligned[-1] - gt_interpolated[-1]))
    summary = {
        "estimate": str(estimate_path),
        "ground_truth": str(ground_truth_path),
        "alignment": alignment,
        "time_offset_s": time_offset,
        "overlap_start_s": float(shifted_times[0]),
        "overlap_end_s": float(shifted_times[-1]),
        "overlap_samples": int(shifted_times.size),
        "alignment_scale": float(scale),
        "path_length_m": path_length,
        "final_drift_m": final_drift,
        "final_drift_percent": (
            100.0 * final_drift / path_length
            if path_length > 1.0e-12
            else None
        ),
        "ate_translation_m": describe(ate),
        "rpe_translation_m": describe(rpe),
        "rpe_delta_s": rpe_delta,
    }
    return summary, shifted_times, ate, rpe


def write_errors(
    path: Path,
    times: Iterable[float],
    ate: Iterable[float],
    rpe: Iterable[float],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["timestamp_s", "timestamp_ns", "ate_translation_m",
             "rpe_translation_m"]
        )
        for timestamp, ate_value, rpe_value in zip(times, ate, rpe):
            writer.writerow(
                [
                    f"{timestamp:.17g}",
                    int(round(timestamp * 1.0e9)),
                    f"{ate_value:.17g}",
                    "" if not math.isfinite(rpe_value)
                    else f"{rpe_value:.17g}",
                ]
            )


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=ESTIMATE_PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=ESTIMATE_PATH")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ATE/RPE evaluation for TUM trajectories"
    )
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument(
        "--run", action="append", required=True, type=parse_run
    )
    parser.add_argument(
        "--align", choices=("se3", "sim3", "none"), default="se3"
    )
    parser.add_argument("--time-offset", type=float, default=0.0)
    parser.add_argument("--rpe-delta", type=float, default=1.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/evaluation")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    try:
        for label, estimate_path in args.run:
            summary, times, ate, rpe = evaluate(
                estimate_path,
                args.gt,
                args.align,
                args.time_offset,
                args.rpe_delta,
            )
            all_summaries[label] = summary
            write_errors(
                args.output_dir / f"{label}_trajectory_errors.csv",
                times,
                ate,
                rpe,
            )
            print(
                f"[{label}] samples={summary['overlap_samples']} "
                f"ATE_RMSE={summary['ate_translation_m']['rmse']:.6f} m "
                f"RPE_RMSE="
                f"{summary['rpe_translation_m']['rmse']} m "
                f"drift={summary['final_drift_percent']}%"
            )
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        print(f"trajectory evaluation error: {error}", file=sys.stderr)
        return 2

    json_path = args.output_dir / "trajectory_summary.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(all_summaries, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"summary={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
