#!/usr/bin/env python3
"""Audit Coco-LIC CASR intervention CSVs without external dependencies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REQUIRED_COLUMNS = {
    "scan_timestamp_s",
    "method_version",
    "state",
    "enabled",
    "apply_to_estimator",
    "eligible",
    "factor_added",
    "applied",
    "data_source_code",
    "route",
    "recovery_rank",
    "requested_activation_strength",
    "used_activation_strength",
    "effective_information_weight",
    "pre_projected_increment_norm",
    "pre_orthogonal_increment_norm",
    "post_projected_increment_norm",
    "post_orthogonal_increment_norm",
    "solver_usable",
    "primary_solver_usable",
    "fallback_attempted",
    "fallback_solver_usable",
    "curvature_matching_enabled",
    "curvature_valid",
    "curvature_tangent_dimension",
    "reference_curvature_max",
    "target_curvature",
    "recovery_curvature_min",
    "recovery_curvature_median",
    "recovery_curvature_max",
    "added_information_min",
    "added_information_median",
    "added_information_max",
    "counterfactual_enabled",
    "counterfactual_solver_usable",
    "counterfactual_total_increment_norm",
    "counterfactual_projected_increment_norm",
    "counterfactual_orthogonal_increment_norm",
    "counterfactual_factor_residual_norm",
    "projected_casr_over_counterfactual",
    "orthogonal_casr_over_counterfactual",
}

INTEGER_COLUMNS = {
    "enabled",
    "apply_to_estimator",
    "eligible",
    "factor_added",
    "applied",
    "data_source_code",
    "solver_usable",
    "primary_solver_usable",
    "fallback_attempted",
    "fallback_solver_usable",
    "curvature_matching_enabled",
    "curvature_valid",
    "counterfactual_enabled",
    "counterfactual_solver_usable",
}

FLOAT_COLUMNS = {
    "scan_timestamp_s",
    "requested_activation_strength",
    "used_activation_strength",
    "effective_information_weight",
    "pre_projected_increment_norm",
    "pre_orthogonal_increment_norm",
    "post_projected_increment_norm",
    "post_orthogonal_increment_norm",
    "reference_curvature_max",
    "target_curvature",
    "recovery_curvature_min",
    "recovery_curvature_median",
    "recovery_curvature_max",
    "added_information_min",
    "added_information_median",
    "added_information_max",
    "counterfactual_total_increment_norm",
    "counterfactual_projected_increment_norm",
    "counterfactual_orthogonal_increment_norm",
    "counterfactual_factor_residual_norm",
    "projected_casr_over_counterfactual",
    "orthogonal_casr_over_counterfactual",
}


def _median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _quantile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _ratios(rows: Iterable[Mapping[str, object]], numerator: str,
            denominator: str) -> List[float]:
    ratios: List[float] = []
    for row in rows:
        den = float(row[denominator])
        num = float(row[numerator])
        if den > 1e-12 and math.isfinite(den) and math.isfinite(num):
            ratios.append(num / den)
    return ratios


def read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"missing columns: {', '.join(missing)}")
        rows: List[Dict[str, object]] = []
        for line_number, raw in enumerate(reader, start=2):
            row: Dict[str, object] = dict(raw)
            try:
                for name in INTEGER_COLUMNS:
                    row[name] = int(str(raw[name]))
                row["recovery_rank"] = int(str(raw["recovery_rank"]))
                row["curvature_tangent_dimension"] = int(
                    str(raw["curvature_tangent_dimension"])
                )
                for name in FLOAT_COLUMNS:
                    row[name] = float(str(raw[name]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"line {line_number}: invalid numeric value: {error}"
                ) from error
            rows.append(row)
    if not rows:
        raise ValueError("CSV contains no data rows")
    return rows


def audit_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = []
    timestamps = [float(row["scan_timestamp_s"]) for row in rows]
    if len(set(timestamps)) != len(timestamps):
        errors.append("scan timestamps are not unique")
    if any(not math.isfinite(value) for value in timestamps):
        errors.append("scan timestamps contain non-finite values")

    methods = Counter(str(row["method_version"]) for row in rows)
    sources = Counter(int(row["data_source_code"]) for row in rows)
    apply_flags = Counter(int(row["apply_to_estimator"]) for row in rows)
    states = Counter(str(row["state"]) for row in rows)
    if set(sources) - {0}:
        errors.append("non-real diagnostics-copy source reached intervention CSV")
    if len(methods) != 1:
        errors.append("multiple intervention method versions are mixed")
    if len(apply_flags) != 1:
        errors.append("apply_to_estimator changed inside one run")

    for index, row in enumerate(rows, start=2):
        state = str(row["state"])
        eligible = int(row["eligible"])
        factor_added = int(row["factor_added"])
        applied = int(row["applied"])
        fallback_attempted = int(row["fallback_attempted"])
        fallback_usable = int(row["fallback_solver_usable"])
        apply_to_estimator = int(row["apply_to_estimator"])
        curvature_matching = int(row["curvature_matching_enabled"])
        curvature_valid = int(row["curvature_valid"])
        for name in FLOAT_COLUMNS - {"scan_timestamp_s"}:
            if not math.isfinite(float(row[name])):
                errors.append(f"line {index}: {name} is non-finite")
        if applied and not factor_added:
            errors.append(f"line {index}: applied=1 but factor_added=0")
        if factor_added and not eligible:
            errors.append(f"line {index}: factor_added=1 but eligible=0")
        if factor_added and not apply_to_estimator:
            errors.append(
                f"line {index}: dry-run row added an estimator factor"
            )
        if state == "dry_run" and (factor_added or applied):
            errors.append(f"line {index}: dry_run has a committed factor")
        if state == "applied" and not (factor_added and applied):
            errors.append(f"line {index}: applied state/flags disagree")
        if factor_added and curvature_matching and not curvature_valid:
            errors.append(
                f"line {index}: curvature-matched factor lacks valid curvature"
            )
        if state in {"curvature_invalid", "curvature_sufficient"} and (
            factor_added or applied
        ):
            errors.append(
                f"line {index}: {state} unexpectedly changed the estimator"
            )
        if state == "solver_failure_recovered":
            if not (factor_added and not applied and fallback_attempted and
                    fallback_usable):
                errors.append(
                    f"line {index}: recovered failure flags disagree"
                )
        if fallback_usable and not fallback_attempted:
            errors.append(
                f"line {index}: fallback usable without an attempted fallback"
            )

    applied_rows = [row for row in rows if int(row["applied"]) == 1]
    measurement_rows = [
        row for row in rows
        if int(row["eligible"]) == 1 or
        str(row["state"]) == "curvature_sufficient"
    ]
    projected_ratios = _ratios(
        measurement_rows,
        "post_projected_increment_norm",
        "pre_projected_increment_norm",
    )
    orthogonal_ratios = _ratios(
        measurement_rows,
        "post_orthogonal_increment_norm",
        "pre_orthogonal_increment_norm",
    )
    if measurement_rows and not projected_ratios:
        warnings.append("measured rows have no nonzero pre projected increment")

    counterfactual_rows = [
        row for row in applied_rows
        if int(row["counterfactual_enabled"]) == 1 and
        int(row["counterfactual_solver_usable"]) == 1
    ]
    projected_counterfactual_ratios = [
        float(row["projected_casr_over_counterfactual"])
        for row in counterfactual_rows
        if math.isfinite(float(row["projected_casr_over_counterfactual"]))
    ]
    orthogonal_counterfactual_ratios = [
        float(row["orthogonal_casr_over_counterfactual"])
        for row in counterfactual_rows
        if math.isfinite(float(row["orthogonal_casr_over_counterfactual"]))
    ]
    added_information = [
        float(row["added_information_median"])
        for row in rows if int(row["curvature_valid"]) == 1
    ]

    return {
        "rows": len(rows),
        "timestamp_start_s": min(timestamps),
        "timestamp_end_s": max(timestamps),
        "method_versions": dict(methods),
        "apply_to_estimator": dict(apply_flags),
        "data_sources": dict(sources),
        "states": dict(states),
        "eligible_rows": sum(int(row["eligible"]) for row in rows),
        "factor_added_rows": sum(int(row["factor_added"]) for row in rows),
        "committed_rows": len(applied_rows),
        "fallback_attempted_rows": sum(
            int(row["fallback_attempted"]) for row in rows
        ),
        "projected_post_over_pre": {
            "count": len(projected_ratios),
            "median": _median(projected_ratios),
            "p90": _quantile(projected_ratios, 0.9),
        },
        "orthogonal_post_over_pre": {
            "count": len(orthogonal_ratios),
            "median": _median(orthogonal_ratios),
            "p90": _quantile(orthogonal_ratios, 0.9),
        },
        "projected_casr_over_same_frame_baseline": {
            "count": len(projected_counterfactual_ratios),
            "median": _median(projected_counterfactual_ratios),
            "p90": _quantile(projected_counterfactual_ratios, 0.9),
        },
        "orthogonal_casr_over_same_frame_baseline": {
            "count": len(orthogonal_counterfactual_ratios),
            "median": _median(orthogonal_counterfactual_ratios),
            "p90": _quantile(orthogonal_counterfactual_ratios, 0.9),
        },
        "median_added_information": {
            "count": len(added_information),
            "median": _median(added_information),
            "p90": _quantile(added_information, 0.9),
        },
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }


def aligned_comparison(
    first: Sequence[Mapping[str, object]],
    second: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    first_by_time = {float(row["scan_timestamp_s"]): row for row in first}
    second_by_time = {float(row["scan_timestamp_s"]): row for row in second}
    common = sorted(set(first_by_time) & set(second_by_time))
    activation_differences = [
        abs(
            float(first_by_time[t]["requested_activation_strength"])
            - float(second_by_time[t]["requested_activation_strength"])
        )
        for t in common
    ]
    route_matches = sum(
        str(first_by_time[t]["route"]) == str(second_by_time[t]["route"])
        for t in common
    )
    rank_matches = sum(
        int(first_by_time[t]["recovery_rank"])
        == int(second_by_time[t]["recovery_rank"])
        for t in common
    )
    return {
        "aligned_rows": len(common),
        "first_only_rows": len(first_by_time) - len(common),
        "second_only_rows": len(second_by_time) - len(common),
        "route_match_fraction": route_matches / len(common) if common else None,
        "rank_match_fraction": rank_matches / len(common) if common else None,
        "activation_absolute_difference_median": _median(
            activation_differences
        ),
        "activation_absolute_difference_p90": _quantile(
            activation_differences, 0.9
        ),
    }


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("use nonempty LABEL=PATH")
    return label, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate CASR intervention safety invariants and summarize "
            "curvature matching and same-frame baseline/CASR changes."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        metavar="LABEL=PATH",
        help="intervention CSV; pass twice to compare dry-run and armed runs",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    args = parser.parse_args()

    loaded: Dict[str, List[Dict[str, object]]] = {}
    output: Dict[str, object] = {"runs": {}}
    try:
        for label, path in args.run:
            if label in loaded:
                raise ValueError(f"duplicate run label: {label}")
            rows = read_csv(path)
            loaded[label] = rows
            output["runs"][label] = audit_rows(rows)  # type: ignore[index]
    except (OSError, ValueError) as error:
        print(f"CASR audit input error: {error}", file=sys.stderr)
        return 2

    labels = list(loaded)
    if len(labels) == 2:
        output["comparison"] = aligned_comparison(
            loaded[labels[0]], loaded[labels[1]]
        )
    elif len(labels) > 2:
        print("CASR audit input error: at most two runs are supported",
              file=sys.stderr)
        return 2

    passed = all(
        bool(summary["passed"])
        for summary in output["runs"].values()  # type: ignore[union-attr]
    )
    output["passed"] = passed
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        for label in labels:
            summary = output["runs"][label]  # type: ignore[index]
            print(f"[{label}] rows={summary['rows']} passed={summary['passed']}")
            print(f"  states={summary['states']}")
            print(
                "  eligible/factor/committed="
                f"{summary['eligible_rows']}/{summary['factor_added_rows']}/"
                f"{summary['committed_rows']}"
            )
            print(
                "  projected post/pre="
                f"{summary['projected_post_over_pre']}"
            )
            print(
                "  orthogonal post/pre="
                f"{summary['orthogonal_post_over_pre']}"
            )
            print(
                "  projected CASR/same-frame-baseline="
                f"{summary['projected_casr_over_same_frame_baseline']}"
            )
            print(
                "  orthogonal CASR/same-frame-baseline="
                f"{summary['orthogonal_casr_over_same_frame_baseline']}"
            )
            print(
                "  curvature-matched added information="
                f"{summary['median_added_information']}"
            )
            for message in summary["errors"]:
                print(f"  ERROR: {message}")
            for message in summary["warnings"]:
                print(f"  WARNING: {message}")
        if "comparison" in output:
            print(f"[comparison] {output['comparison']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
