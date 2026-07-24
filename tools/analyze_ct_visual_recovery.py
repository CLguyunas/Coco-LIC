#!/usr/bin/env python3
"""Audit CT weak-direction-driven visual complement CSV files."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REQUIRED_FIELDS = {
    "scan_time_ns",
    "image_time_ns",
    "valid",
    "eligible",
    "state",
    "apply_requested",
    "applied",
    "weak_rank",
    "visual_candidate_count",
    "valid_factor_count",
    "rejected_factor_count",
    "baseline_selected_count",
    "eligible_unselected_count",
    "additional_budget",
    "additional_selected_count",
    "weak_d_efficiency_target",
    "baseline_weak_d_efficiency",
    "repaired_weak_d_efficiency",
    "baseline_weak_min_retention",
    "repaired_weak_min_retention",
    "baseline_global_d_efficiency",
    "repaired_global_d_efficiency",
    "median_factorization_error",
    "max_factorization_error",
}

RATIO_FIELDS = (
    "weak_d_efficiency_target",
    "baseline_weak_d_efficiency",
    "repaired_weak_d_efficiency",
    "baseline_weak_min_retention",
    "repaired_weak_min_retention",
    "baseline_global_d_efficiency",
    "repaired_global_d_efficiency",
)

TOLERANCE = 1.0e-9
FACTORIZATION_TOLERANCE = 1.01e-5


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=CSV_PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=CSV_PATH")
    return label, Path(path)


def percentile(values: Iterable[float], probability: float) -> Optional[float]:
    ordered = sorted(values)
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
    }


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_FIELDS - fields)
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        return list(reader)


def as_int(row: Dict[str, str], field: str) -> int:
    return int(row[field])


def as_float(row: Dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}={row[field]}")
    return value


def audit(label: str, path: Path) -> bool:
    rows = read_rows(path)
    errors: List[str] = []
    states = Counter(row["state"] for row in rows)
    valid_rows: List[Dict[str, str]] = []
    eligible_rows: List[Dict[str, str]] = []
    applied_rows: List[Dict[str, str]] = []

    for row_index, row in enumerate(rows, start=2):
        try:
            valid = as_int(row, "valid") == 1
            eligible = as_int(row, "eligible") == 1
            apply_requested = as_int(row, "apply_requested") == 1
            applied = as_int(row, "applied") == 1
            candidates = as_int(row, "visual_candidate_count")
            valid_factors = as_int(row, "valid_factor_count")
            rejected_factors = as_int(row, "rejected_factor_count")
            unselected = as_int(row, "eligible_unselected_count")
            budget = as_int(row, "additional_budget")
            additional = as_int(row, "additional_selected_count")

            if min(
                candidates,
                valid_factors,
                rejected_factors,
                unselected,
                budget,
                additional,
            ) < 0:
                errors.append(f"line {row_index}: negative count")
            factors_were_evaluated = (
                valid
                or valid_factors > 0
                or rejected_factors > 0
                or row["state"] == "no_factor_consistent_visual"
            )
            if (
                factors_were_evaluated
                and valid_factors + rejected_factors != candidates
            ):
                errors.append(
                    f"line {row_index}: factor accounting mismatch"
                )
            if additional > min(unselected, budget):
                errors.append(
                    f"line {row_index}: additional selection exceeds budget"
                )
            if applied and (not apply_requested or additional == 0):
                errors.append(
                    f"line {row_index}: invalid applied state"
                )
            if apply_requested and additional > 0 and valid and not applied:
                errors.append(
                    f"line {row_index}: armed selection was not applied"
                )

            if valid:
                valid_rows.append(row)
                for field in RATIO_FIELDS:
                    value = as_float(row, field)
                    if value < -TOLERANCE or value > 1.0 + TOLERANCE:
                        errors.append(
                            f"line {row_index}: {field} outside [0,1]"
                        )
                baseline_weak = as_float(
                    row, "baseline_weak_d_efficiency"
                )
                repaired_weak = as_float(
                    row, "repaired_weak_d_efficiency"
                )
                baseline_min = as_float(
                    row, "baseline_weak_min_retention"
                )
                repaired_min = as_float(
                    row, "repaired_weak_min_retention"
                )
                baseline_global = as_float(
                    row, "baseline_global_d_efficiency"
                )
                repaired_global = as_float(
                    row, "repaired_global_d_efficiency"
                )
                if repaired_weak + TOLERANCE < baseline_weak:
                    errors.append(
                        f"line {row_index}: weak D-efficiency decreased"
                    )
                if repaired_min + TOLERANCE < baseline_min:
                    errors.append(
                        f"line {row_index}: weak minimum retention decreased"
                    )
                if repaired_global + TOLERANCE < baseline_global:
                    errors.append(
                        f"line {row_index}: global D-efficiency decreased"
                    )
                if row["state"] == "target_reached":
                    target = as_float(row, "weak_d_efficiency_target")
                    if repaired_weak + TOLERANCE < target:
                        errors.append(
                            f"line {row_index}: target state below target"
                        )
                maximum_error = as_float(
                    row, "max_factorization_error"
                )
                if maximum_error > FACTORIZATION_TOLERANCE:
                    errors.append(
                        f"line {row_index}: factorization error too large"
                    )

            if eligible:
                eligible_rows.append(row)
            if applied:
                applied_rows.append(row)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"line {row_index}: {exc}")

    passed = bool(rows) and not errors

    def values(source: Iterable[Dict[str, str]], field: str) -> List[float]:
        return [float(row[field]) for row in source]

    factor_acceptance = [
        float(row["valid_factor_count"])
        / float(row["visual_candidate_count"])
        for row in eligible_rows
        if int(row["visual_candidate_count"]) > 0
    ]
    before_target = 0
    after_target = 0
    for row in valid_rows:
        target = float(row["weak_d_efficiency_target"])
        before_target += (
            float(row["baseline_weak_d_efficiency"]) >= target
        )
        after_target += (
            float(row["repaired_weak_d_efficiency"]) >= target
        )

    print(f"[{label}] rows={len(rows)} passed={passed}")
    print(
        "  states="
        f"{dict(states)} valid/eligible/applied="
        f"{len(valid_rows)}/{len(eligible_rows)}/{len(applied_rows)}"
    )
    print(
        "  weak-rank="
        f"{describe(values(valid_rows, 'weak_rank'))}"
    )
    print(
        "  additional-selected="
        f"{describe(values(valid_rows, 'additional_selected_count'))}"
    )
    print(
        "  factor-acceptance="
        f"{describe(factor_acceptance)}"
    )
    print(
        "  weak D-efficiency baseline/repaired="
        f"{describe(values(valid_rows, 'baseline_weak_d_efficiency'))}/"
        f"{describe(values(valid_rows, 'repaired_weak_d_efficiency'))}"
    )
    print(
        "  weak minimum retention baseline/repaired="
        f"{describe(values(valid_rows, 'baseline_weak_min_retention'))}/"
        f"{describe(values(valid_rows, 'repaired_weak_min_retention'))}"
    )
    print(
        "  global D-efficiency baseline/repaired="
        f"{describe(values(valid_rows, 'baseline_global_d_efficiency'))}/"
        f"{describe(values(valid_rows, 'repaired_global_d_efficiency'))}"
    )
    print(
        "  weak-target reached before/after="
        f"{before_target}/{after_target} of {len(valid_rows)}"
    )
    print(
        "  factorization error median/max="
        f"{describe(values(valid_rows, 'median_factorization_error'))}/"
        f"{describe(values(valid_rows, 'max_factorization_error'))}"
    )
    if errors:
        for error in errors[:20]:
            print(f"  ERROR: {error}")
        if len(errors) > 20:
            print(f"  ERROR: ... {len(errors) - 20} more")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check structural safety and summarize CT-directed visual "
            "recovery. A pass is an implementation audit, not a trajectory "
            "accuracy claim."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run,
        metavar="LABEL=CSV_PATH",
    )
    args = parser.parse_args()

    all_passed = True
    try:
        for label, path in args.run:
            all_passed = audit(label, path) and all_passed
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"CT visual recovery audit input error: {exc}", file=sys.stderr)
        return 2
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
