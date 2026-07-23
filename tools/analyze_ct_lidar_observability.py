#!/usr/bin/env python3
"""Audit the read-only continuous-time LiDAR observability CSV."""

import argparse
import csv
import math
import statistics
import sys
from collections import Counter
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def active_runs(rows):
    runs = []
    start = None
    for index, row in enumerate(rows):
        active = row["persistent_degenerate"] == "1"
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(rows) - 1))
    return runs


def audit(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("CSV contains no data rows")

    valid_rows = [row for row in rows if row["valid"] == "1"]
    invalid_rows = len(rows) - len(valid_rows)
    states = Counter(row["state"] for row in rows)
    min_ratios = []
    weak_ranks = []
    factor_counts = []
    ordering_failures = 0
    rank_failures = 0

    for row in valid_rows:
        ratios = [float(row[f"relative_{index}"]) for index in range(6)]
        if not all(math.isfinite(value) for value in ratios):
            raise ValueError("non-finite relative eigenvalue")
        if any(ratios[index] > ratios[index + 1] + 1.0e-10
               for index in range(5)):
            ordering_failures += 1
        reported_rank = int(row["weak_rank"])
        threshold = float(row["decision_threshold"])
        inferred_rank = sum(value < threshold for value in ratios)
        if reported_rank != inferred_rank:
            rank_failures += 1
        min_ratios.append(ratios[0])
        weak_ranks.append(reported_rank)
        factor_counts.append(int(row["evaluated_factor_count"]))

    runs = active_runs(rows)
    longest_run = max((end - start + 1 for start, end in runs), default=0)
    summary = {
        "rows": len(rows),
        "valid_rows": len(valid_rows),
        "invalid_rows": invalid_rows,
        "states": dict(states),
        "persistent_runs": runs,
        "longest_persistent_run": longest_run,
        "ordering_failures": ordering_failures,
        "rank_consistency_failures": rank_failures,
        "minimum_relative_eigenvalue": {
            "median": statistics.median(min_ratios) if min_ratios else None,
            "p10": percentile(min_ratios, 0.10),
            "p90": percentile(min_ratios, 0.90),
        },
        "weak_rank": {
            "median": statistics.median(weak_ranks) if weak_ranks else None,
            "max": max(weak_ranks) if weak_ranks else None,
        },
        "evaluated_factor_count": {
            "median": statistics.median(factor_counts) if factor_counts else None,
            "min": min(factor_counts) if factor_counts else None,
        },
    }
    passed = bool(valid_rows) and ordering_failures == 0 and rank_failures == 0
    return passed, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="label and CT observability CSV path; may be repeated",
    )
    arguments = parser.parse_args()

    all_passed = True
    for item in arguments.run:
        if "=" not in item:
            parser.error("--run must use LABEL=CSV")
        label, raw_path = item.split("=", 1)
        try:
            passed, summary = audit(Path(raw_path))
        except (OSError, ValueError, KeyError) as error:
            print(f"[{label}] input error: {error}", file=sys.stderr)
            all_passed = False
            continue
        all_passed = all_passed and passed
        print(f"[{label}] passed={passed}")
        for key, value in summary.items():
            print(f"  {key}={value}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
