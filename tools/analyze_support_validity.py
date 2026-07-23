#!/usr/bin/env python3
"""Audit whether natural spline-support alarms correspond to real LIC weakness.

This script does not treat a low support-quality score as proof.  It aligns the
read-only timestamp-support diagnostic with an independent projection of the
actual final-LIC Hessian onto the diagnosed support basis, then reports whether
natural persistent alarms are accompanied by missing optimizer information.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


OBS_REQUIRED = {
    "scan_timestamp_s",
    "support_valid",
    "support_quality_min",
    "support_degenerate_state",
    "degeneracy_cause",
    "support_sample_num",
    "support_empty_interval_num",
    "support_occupied_temporal_bin_ratio",
    "support_temporal_mass_l1",
}

INTERVENTION_REQUIRED = {
    "scan_timestamp_s",
    "support_curvature_audit_valid",
    "support_curvature_rank",
    "support_curvature_min_over_reference_max",
    "support_curvature_mean_over_reference_mean",
    "support_curvature_below_target_fraction",
}


def load_csv(path: Path, required: Iterable[str]) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = sorted(set(required) - columns)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        return list(reader)


def as_float(row: Mapping[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}={row[key]!r}")
    return value


def as_int(row: Mapping[str, str], key: str) -> int:
    return int(float(row[key]))


def quantiles(values: Sequence[float]) -> Mapping[str, Optional[float]]:
    if not values:
        return {"count": 0, "median": None, "p10": None, "p90": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "median": statistics.median(ordered),
        "p10": percentile(0.10),
        "p90": percentile(0.90),
    }


def average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        for offset in range(start, end):
            ranks[order[offset]] = rank
        start = end
    return ranks


def pearson(lhs: Sequence[float], rhs: Sequence[float]) -> Optional[float]:
    if len(lhs) != len(rhs) or len(lhs) < 3:
        return None
    lhs_mean = statistics.fmean(lhs)
    rhs_mean = statistics.fmean(rhs)
    numerator = sum((x - lhs_mean) * (y - rhs_mean) for x, y in zip(lhs, rhs))
    lhs_energy = sum((x - lhs_mean) ** 2 for x in lhs)
    rhs_energy = sum((y - rhs_mean) ** 2 for y in rhs)
    denominator = math.sqrt(lhs_energy * rhs_energy)
    if denominator <= 0.0:
        return None
    return numerator / denominator


def spearman(lhs: Sequence[float], rhs: Sequence[float]) -> Optional[float]:
    return pearson(average_ranks(lhs), average_ranks(rhs))


def true_runs(states: Sequence[bool]) -> List[int]:
    runs: List[int] = []
    length = 0
    for state in states:
        if state:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    if length:
        runs.append(length)
    return runs


def timestamp_key(row: Mapping[str, str]) -> int:
    return int(round(as_float(row, "scan_timestamp_s") * 1.0e6))


def audit(label: str, observability_path: Path,
          intervention_path: Path) -> Mapping[str, object]:
    observability = load_csv(observability_path, OBS_REQUIRED)
    intervention = load_csv(intervention_path, INTERVENTION_REQUIRED)
    intervention_by_time = {timestamp_key(row): row for row in intervention}

    aligned: List[Tuple[Dict[str, str], Dict[str, str]]] = []
    for row in observability:
        other = intervention_by_time.get(timestamp_key(row))
        if other is not None:
            aligned.append((row, other))

    support_valid = [pair for pair in aligned if as_int(pair[0], "support_valid") == 1]
    audited = [
        pair for pair in support_valid
        if as_int(pair[1], "support_curvature_audit_valid") == 1
    ]
    active_flags = [
        as_int(row, "support_degenerate_state") == 1 for row, _ in aligned
    ]
    active = [pair for pair in audited if as_int(pair[0], "support_degenerate_state") == 1]
    healthy = [pair for pair in audited if as_int(pair[0], "support_degenerate_state") == 0]
    causes = Counter(as_int(row, "degeneracy_cause") for row, _ in aligned)

    quality = [as_float(row, "support_quality_min") for row, _ in audited]
    curvature_min = [
        as_float(other, "support_curvature_min_over_reference_max")
        for _, other in audited
    ]
    curvature_mean = [
        as_float(other, "support_curvature_mean_over_reference_mean")
        for _, other in audited
    ]
    temporal_l1 = [as_float(row, "support_temporal_mass_l1") for row, _ in audited]
    occupied = [
        as_float(row, "support_occupied_temporal_bin_ratio")
        for row, _ in audited
    ]

    active_below = [
        as_float(other, "support_curvature_below_target_fraction")
        for _, other in active
    ]
    active_runs = true_runs(active_flags)
    audit_coverage = len(audited) / len(support_valid) if support_valid else 0.0

    if not support_valid:
        verdict = "NO_VALID_SUPPORT_DIAGNOSTIC"
    elif not active_runs:
        verdict = "NO_NATURAL_SUPPORT_TRIGGER"
    elif audit_coverage < 0.90:
        verdict = "INSUFFICIENT_LIC_CURVATURE_COVERAGE"
    elif not active:
        verdict = "TRIGGER_NOT_ALIGNED_WITH_CURVATURE_LOG"
    elif statistics.median(active_below) >= 0.50:
        verdict = "NATURAL_SUPPORT_WEAKNESS_CANDIDATE"
    else:
        verdict = "SUPPORT_ALARM_NOT_CONFIRMED_BY_LIC_CURVATURE"

    def select(pairs: Sequence[Tuple[Dict[str, str], Dict[str, str]]], key: str) -> List[float]:
        return [as_float(other, key) for _, other in pairs]

    return {
        "label": label,
        "rows": {"observability": len(observability), "intervention": len(intervention),
                 "aligned": len(aligned)},
        "support": {
            "valid": len(support_valid),
            "active": sum(active_flags),
            "support_only": causes.get(2, 0),
            "coupled": causes.get(3, 0),
            "active_runs": active_runs,
            "longest_active_run": max(active_runs, default=0),
        },
        "curvature_audit_coverage": audit_coverage,
        "all_quality": quantiles(quality),
        "all_temporal_mass_l1": quantiles(temporal_l1),
        "all_occupied_bin_ratio": quantiles(occupied),
        "healthy_min_over_reference_max": quantiles(select(
            healthy, "support_curvature_min_over_reference_max")),
        "active_min_over_reference_max": quantiles(select(
            active, "support_curvature_min_over_reference_max")),
        "healthy_mean_over_reference_mean": quantiles(select(
            healthy, "support_curvature_mean_over_reference_mean")),
        "active_mean_over_reference_mean": quantiles(select(
            active, "support_curvature_mean_over_reference_mean")),
        "active_below_target_fraction": quantiles(active_below),
        "spearman": {
            "quality_vs_min_curvature": spearman(quality, curvature_min),
            "quality_vs_mean_curvature": spearman(quality, curvature_mean),
            "temporal_mass_l1_vs_quality": spearman(temporal_l1, quality),
            "occupied_bins_vs_quality": spearman(occupied, quality),
        },
        "verdict": verdict,
    }


def parse_run(value: str) -> Tuple[str, Path, Path]:
    if "=" not in value or "," not in value:
        raise argparse.ArgumentTypeError(
            "expected LABEL=OBSERVABILITY.csv,INTERVENTION.csv")
    label, paths = value.split("=", 1)
    observability, intervention = paths.split(",", 1)
    if not label or not observability or not intervention:
        raise argparse.ArgumentTypeError(
            "expected LABEL=OBSERVABILITY.csv,INTERVENTION.csv")
    return label, Path(observability), Path(intervention)


def print_report(report: Mapping[str, object]) -> None:
    print(f"[{report['label']}] verdict={report['verdict']}")
    print(f"  rows={report['rows']}")
    print(f"  support={report['support']}")
    print(f"  curvature-audit-coverage={report['curvature_audit_coverage']:.3f}")
    print(f"  healthy min/max={report['healthy_min_over_reference_max']}")
    print(f"  active min/max={report['active_min_over_reference_max']}")
    print(f"  healthy mean/mean={report['healthy_mean_over_reference_mean']}")
    print(f"  active mean/mean={report['active_mean_over_reference_mean']}")
    print(f"  active below-target fraction={report['active_below_target_fraction']}")
    print(f"  correlations={report['spearman']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, type=parse_run,
        metavar="LABEL=OBSERVABILITY.csv,INTERVENTION.csv")
    args = parser.parse_args(argv)
    try:
        for label, observability, intervention in args.run:
            print_report(audit(label, observability, intervention))
    except (OSError, ValueError) as error:
        print(f"support validity audit error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
