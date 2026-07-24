#!/usr/bin/env python3
"""Aggregate repeated trajectory evaluations into paper-ready CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_manifest(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"sequence", "method", "repeat", "summary_json", "run_label"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    records = []
    for row in rows:
        summary_path = Path(row["summary_json"])
        with summary_path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        run = summary[row["run_label"]]
        records.append(
            {
                "sequence": row["sequence"],
                "method": row["method"],
                "repeat": row["repeat"],
                "ate_rmse": run["ate_translation_m"]["rmse"],
                "rpe_rmse": run["rpe_translation_m"]["rmse"],
                "drift_percent": run["final_drift_percent"],
            }
        )
    return records


def mean_std(values):
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None, None
    return statistics.fmean(finite), (
        statistics.stdev(finite) if len(finite) > 1 else 0.0
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline", default="Original")
    parser.add_argument(
        "--output", type=Path, default=Path("data/evaluation/repeat_summary.csv")
    )
    args = parser.parse_args()
    try:
        records = load_manifest(args.manifest)
        grouped = defaultdict(list)
        for record in records:
            grouped[(record["sequence"], record["method"])].append(record)
        baseline_means = {}
        for (sequence, method), group in grouped.items():
            if method == args.baseline:
                baseline_means[sequence] = mean_std(
                    record["ate_rmse"] for record in group
                )[0]

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "sequence",
                    "method",
                    "repeats",
                    "ate_rmse_mean_m",
                    "ate_rmse_std_m",
                    "rpe_rmse_mean_m",
                    "rpe_rmse_std_m",
                    "drift_mean_percent",
                    "drift_std_percent",
                    f"ate_improvement_vs_{args.baseline}_percent",
                ]
            )
            for (sequence, method), group in sorted(grouped.items()):
                ate_mean, ate_std = mean_std(
                    record["ate_rmse"] for record in group
                )
                rpe_mean, rpe_std = mean_std(
                    record["rpe_rmse"] for record in group
                )
                drift_mean, drift_std = mean_std(
                    record["drift_percent"] for record in group
                )
                baseline = baseline_means.get(sequence)
                improvement = (
                    100.0 * (baseline - ate_mean) / baseline
                    if baseline and ate_mean is not None
                    else None
                )
                writer.writerow(
                    [
                        sequence,
                        method,
                        len(group),
                        ate_mean,
                        ate_std,
                        rpe_mean,
                        rpe_std,
                        drift_mean,
                        drift_std,
                        improvement,
                    ]
                )
        print(f"summary={args.output}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"repeat summary input error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
