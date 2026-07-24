#!/usr/bin/env python3
"""Summarize per-frame QI-CTDR runtime and observation counts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


TIME_FIELDS = (
    "ct_detector_ms",
    "qi_lidar_ms",
    "qi_visual_ms",
    "ct_recovery_ms",
    "lic_solver_ms",
    "core_total_ms",
)
COUNT_FIELDS = (
    "lidar_candidates",
    "lidar_selected",
    "visual_candidates",
    "visual_selected",
    "recovery_added",
)


def percentile(values, probability) -> Optional[float]:
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
        "mean": statistics.fmean(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "p90": percentile(finite, 0.90),
        "max": max(finite) if finite else None,
    }


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PROFILE_CSV")
    label, path = value.split("=", 1)
    return label, Path(path)


def audit(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = set(TIME_FIELDS) | set(COUNT_FIELDS)
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")
    successful = [row for row in rows if row["optimization_success"] == "1"]
    source = successful if successful else rows
    summary = {
        "rows": len(rows),
        "successful_rows": len(successful),
        "timing_ms": {
            field: describe(float(row[field]) for row in source)
            for field in TIME_FIELDS
        },
        "counts": {
            field: describe(float(row[field]) for row in source)
            for field in COUNT_FIELDS
        },
    }
    total_solver = sum(float(row["lic_solver_ms"]) for row in source)
    total_modules = sum(
        float(row["ct_detector_ms"])
        + float(row["qi_lidar_ms"])
        + float(row["qi_visual_ms"])
        + float(row["ct_recovery_ms"])
        for row in source
    )
    summary["module_to_solver_time_percent"] = (
        100.0 * total_modules / total_solver if total_solver > 0.0 else None
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", required=True, type=parse_run
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    summaries = {}
    try:
        for label, path in args.run:
            summary = audit(path)
            summaries[label] = summary
            print(f"[{label}]")
            for key, value in summary.items():
                print(f"  {key}={value}")
        if len(summaries) >= 2:
            labels = list(summaries)
            baseline = summaries[labels[0]]["timing_ms"]["core_total_ms"]["mean"]
            if baseline and baseline > 0.0:
                comparison = {}
                for label in labels[1:]:
                    mean = summaries[label]["timing_ms"]["core_total_ms"]["mean"]
                    comparison[label] = {
                        "core_time_overhead_percent": (
                            100.0 * (mean / baseline - 1.0)
                            if mean is not None
                            else None
                        )
                    }
                summaries["comparison_to_first_run"] = comparison
                print(f"[comparison_to_{labels[0]}] {comparison}")
    except (OSError, ValueError, KeyError) as error:
        print(f"profile input error: {error}", file=sys.stderr)
        return 2
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as stream:
            json.dump(summaries, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
