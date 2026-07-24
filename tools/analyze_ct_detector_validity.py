#!/usr/bin/env python3
"""Joint audit of CT detector output and trajectory error.

This tool does not treat trajectory error as a perfect degeneracy label.
Instead it measures whether low CT information and persistent activations
concentrate independently evaluated local trajectory error, and it can compare
the proposed continuous-time detector with the scan-6DoF ablation.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def percentile(values: Sequence[float], probability: float) -> Optional[float]:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return None
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: Iterable[float]) -> Dict[str, Optional[float]]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "count": len(finite),
        "median": statistics.median(finite) if finite else None,
        "p10": percentile(finite, 0.10),
        "p90": percentile(finite, 0.90),
        "mean": statistics.fmean(finite) if finite else None,
    }


def ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    output = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = 0.5 * (cursor + end - 1)
        for position in range(cursor, end):
            output[order[position]] = rank
        cursor = end
    return output


def correlation(first: Sequence[float], second: Sequence[float]) -> Optional[float]:
    pairs = [
        (x_value, y_value)
        for x_value, y_value in zip(first, second)
        if math.isfinite(x_value) and math.isfinite(y_value)
    ]
    if len(pairs) < 3:
        return None
    x_values, y_values = zip(*pairs)
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in pairs
    )
    x_energy = sum((value - x_mean) ** 2 for value in x_values)
    y_energy = sum((value - y_mean) ** 2 for value in y_values)
    denominator = math.sqrt(x_energy * y_energy)
    return numerator / denominator if denominator > 0.0 else None


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / float(len(positives) * len(negatives))


def read_detector(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "reference_time_ns",
        "valid",
        "persistent_degenerate",
        "weak_rank",
        "relative_0",
    }
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")
    return rows


def read_errors(path: Path, field: str) -> Tuple[List[int], List[float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    timestamps = []
    errors = []
    for row in rows:
        raw_error = row.get(field, "")
        if not raw_error:
            continue
        value = float(raw_error)
        if not math.isfinite(value):
            continue
        timestamps.append(int(row["timestamp_ns"]))
        errors.append(value)
    order = sorted(range(len(timestamps)), key=timestamps.__getitem__)
    return [timestamps[index] for index in order], [
        errors[index] for index in order
    ]


def nearest_error(
    timestamp: int,
    error_times: Sequence[int],
    errors: Sequence[float],
    maximum_delta_ns: int,
) -> Optional[float]:
    position = bisect.bisect_left(error_times, timestamp)
    candidates = []
    if position < len(error_times):
        candidates.append(position)
    if position > 0:
        candidates.append(position - 1)
    if not candidates:
        return None
    best = min(candidates, key=lambda index: abs(error_times[index] - timestamp))
    if abs(error_times[best] - timestamp) > maximum_delta_ns:
        return None
    return errors[best]


def persistent_runs(active: Sequence[bool]) -> List[int]:
    lengths = []
    start = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            lengths.append(index - start)
            start = None
    if start is not None:
        lengths.append(len(active) - start)
    return lengths


def audit(
    detector_path: Path,
    errors_path: Path,
    error_field: str,
    maximum_delta_ns: int,
):
    rows = read_detector(detector_path)
    error_times, errors = read_errors(errors_path, error_field)
    samples = []
    for row in rows:
        if row["valid"] != "1":
            continue
        error = nearest_error(
            int(row["reference_time_ns"]),
            error_times,
            errors,
            maximum_delta_ns,
        )
        if error is None:
            continue
        minimum_ratio = float(row["relative_0"])
        if minimum_ratio <= 0.0 or not math.isfinite(minimum_ratio):
            continue
        samples.append(
            {
                "time": int(row["reference_time_ns"]),
                "active": row["persistent_degenerate"] == "1",
                "rank": int(row["weak_rank"]),
                "ratio": minimum_ratio,
                "error": error,
            }
        )
    if not samples:
        raise ValueError("no detector/error rows could be aligned")

    all_errors = [sample["error"] for sample in samples]
    high_error_threshold = percentile(all_errors, 0.80)
    high_error = [
        sample["error"] >= high_error_threshold for sample in samples
    ]
    active = [sample["active"] for sample in samples]
    true_positive = sum(a and h for a, h in zip(active, high_error))
    active_count = sum(active)
    high_count = sum(high_error)
    precision = true_positive / active_count if active_count else None
    recall = true_positive / high_count if high_count else None
    active_errors = [
        sample["error"] for sample in samples if sample["active"]
    ]
    healthy_errors = [
        sample["error"] for sample in samples if not sample["active"]
    ]
    active_median = statistics.median(active_errors) if active_errors else None
    healthy_median = (
        statistics.median(healthy_errors) if healthy_errors else None
    )
    mode = rows[0].get("detector_mode", "legacy_continuous_time")
    summary = {
        "detector_mode": mode,
        "aligned_samples": len(samples),
        "activation_fraction": active_count / len(samples),
        "persistent_run_count": len(persistent_runs(active)),
        "longest_persistent_run": max(persistent_runs(active), default=0),
        "active_error": describe(active_errors),
        "healthy_error": describe(healthy_errors),
        "active_to_healthy_median_error_ratio": (
            active_median / healthy_median
            if active_median is not None
            and healthy_median is not None
            and healthy_median > 0.0
            else None
        ),
        "high_error_definition": f"top_20_percent_{error_field}",
        "high_error_threshold": high_error_threshold,
        "active_high_error_precision": precision,
        "active_high_error_recall": recall,
        "low_information_high_error_auroc": auroc(
            [-math.log10(sample["ratio"]) for sample in samples],
            high_error,
        ),
        "spearman_log_ratio_vs_error": correlation(
            ranks([math.log10(sample["ratio"]) for sample in samples]),
            ranks(all_errors),
        ),
    }
    active_times = {sample["time"] for sample in samples if sample["active"]}
    return summary, active_times


def parse_run(value: str) -> Tuple[str, Path, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "expected LABEL=DETECTOR_CSV,TRAJECTORY_ERROR_CSV"
        )
    label, paths = value.split("=", 1)
    parts = paths.split(",")
    if not label or len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "expected LABEL=DETECTOR_CSV,TRAJECTORY_ERROR_CSV"
        )
    return label, Path(parts[0]), Path(parts[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", required=True, type=parse_run
    )
    parser.add_argument(
        "--error-field",
        choices=("ate_translation_m", "rpe_translation_m"),
        default="rpe_translation_m",
    )
    parser.add_argument("--max-time-difference", type=float, default=0.05)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    summaries = {}
    active_sets = {}
    try:
        for label, detector_path, errors_path in args.run:
            summary, active_times = audit(
                detector_path,
                errors_path,
                args.error_field,
                int(args.max_time_difference * 1.0e9),
            )
            summaries[label] = summary
            active_sets[label] = active_times
            print(f"[{label}] mode={summary['detector_mode']}")
            for key, value in summary.items():
                if key != "detector_mode":
                    print(f"  {key}={value}")
        if len(active_sets) == 2:
            first_label, second_label = active_sets.keys()
            first = active_sets[first_label]
            second = active_sets[second_label]
            union = first | second
            intersection = first & second
            comparison = {
                "labels": [first_label, second_label],
                "active_jaccard": (
                    len(intersection) / len(union) if union else 1.0
                ),
                f"{first_label}_only": len(first - second),
                f"{second_label}_only": len(second - first),
            }
            summaries["comparison"] = comparison
            print(f"[comparison] {comparison}")
    except (OSError, ValueError, KeyError) as error:
        print(f"detector validity input error: {error}", file=sys.stderr)
        return 2

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as stream:
            json.dump(summaries, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
