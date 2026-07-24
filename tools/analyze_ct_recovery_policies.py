#!/usr/bin/env python3
"""Compare weak-subspace, Random, and Global-D recovery policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


POLICIES = ("weak_policy", "random_policy", "global_d_policy")
METRICS = (
    "weak_d_efficiency",
    "weak_min_retention",
    "global_d_efficiency",
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
        "p10": percentile(finite, 0.10),
        "p90": percentile(finite, 0.90),
    }


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=RECOVERY_CSV")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=RECOVERY_CSV")
    return label, Path(path)


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "valid",
        "eligible",
        "selection_policy",
        "baseline_selected_count",
        "additional_selected_count",
        "weak_policy_selected_count",
    }
    for policy in POLICIES:
        for metric in METRICS:
            required.add(f"{policy}_{metric}")
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}")
    return rows


def audit(path: Path):
    rows = read_rows(path)
    valid = [row for row in rows if row["valid"] == "1"]
    eligible = [row for row in valid if row["eligible"] == "1"]
    repaired = [
        row for row in eligible if int(row["weak_policy_selected_count"]) > 0
    ]
    count_mismatches = [
        row for row in repaired
        if int(row["additional_selected_count"])
        != int(row["weak_policy_selected_count"])
    ]

    policy_summary = {}
    for policy in POLICIES:
        policy_summary[policy] = {
            metric: describe(
                float(row[f"{policy}_{metric}"]) for row in repaired
            )
            for metric in METRICS
        }

    def win_fraction(first: str, second: str, metric: str):
        wins = 0
        ties = 0
        for row in repaired:
            first_value = float(row[f"{first}_{metric}"])
            second_value = float(row[f"{second}_{metric}"])
            if first_value > second_value + 1.0e-12:
                wins += 1
            elif abs(first_value - second_value) <= 1.0e-12:
                ties += 1
        return {
            "count": len(repaired),
            "strict_win_fraction": wins / len(repaired) if repaired else None,
            "tie_fraction": ties / len(repaired) if repaired else None,
        }

    baseline_selected = sum(
        int(row["baseline_selected_count"]) for row in eligible
    )
    additional = sum(
        int(row["weak_policy_selected_count"]) for row in eligible
    )
    policies_in_file = sorted(
        {row["selection_policy"] for row in rows}
    )
    summary = {
        "rows": len(rows),
        "valid_rows": len(valid),
        "eligible_rows": len(eligible),
        "repair_rows": len(repaired),
        "selection_policies": policies_in_file,
        "matched_count_failures": len(count_mismatches),
        "additional_observation_overhead_percent": (
            100.0 * additional / baseline_selected
            if baseline_selected
            else None
        ),
        "additional_count": describe(
            int(row["weak_policy_selected_count"]) for row in repaired
        ),
        "policy_metrics": policy_summary,
        "weak_vs_random": {
            metric: win_fraction("weak_policy", "random_policy", metric)
            for metric in METRICS
        },
        "weak_vs_global_d": {
            metric: win_fraction("weak_policy", "global_d_policy", metric)
            for metric in METRICS
        },
    }
    passed = bool(rows) and not count_mismatches
    return passed, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", required=True, type=parse_run
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    summaries = {}
    all_passed = True
    try:
        for label, path in args.run:
            passed, summary = audit(path)
            all_passed = passed and all_passed
            summaries[label] = summary
            print(f"[{label}] passed={passed}")
            for key, value in summary.items():
                print(f"  {key}={value}")
    except (OSError, ValueError, KeyError) as error:
        print(f"recovery policy input error: {error}", file=sys.stderr)
        return 2
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as stream:
            json.dump(summaries, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
