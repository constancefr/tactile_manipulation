#!/usr/bin/env python3
"""Rank tactile-band detector parameters against a labelled image directory.

The default dataset contains one rectangular tactile feature per image, so a
prediction is correct when the detector finds one band (two boundary edges).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from detect_tactile_bands import (
    angle_difference,
    cluster_segments,
    dominant_angle,
    find_segments,
    image_paths,
    preprocess,
    select_edges,
)


DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1] / "Data" / "Single_rect_Data"
)


@dataclass(frozen=True)
class Parameters:
    border_fraction: float
    min_line_fraction: float
    max_gap_fraction: float
    hough_threshold: int
    angle_tolerance: float
    cluster_distance: float
    interval_gap: float
    min_support_fraction: float
    relative_score: float
    max_edges: int


def comma_separated_floats(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values:
        raise argparse.ArgumentTypeError("provide at least one number")
    return values


def comma_separated_ints(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values:
        raise argparse.ArgumentTypeError("provide at least one integer")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep detector parameters and rank them by exact band-count accuracy."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <input-dir>/hyperparameter_sweep.",
    )
    parser.add_argument(
        "--expected-bands",
        type=int,
        default=1,
        help="Correct feature count per image (default: 1, equivalent to 2 edges).",
    )
    parser.add_argument("--top", type=int, default=10, help="Rows to print.")

    parser.add_argument(
        "--border-fractions", type=comma_separated_floats, default=[0.04]
    )
    parser.add_argument(
        "--min-line-fractions",
        type=comma_separated_floats,
        default=[0.18, 0.22],
    )
    parser.add_argument(
        "--max-gap-fractions",
        type=comma_separated_floats,
        default=[0.02, 0.04, 0.05, 0.06],
    )
    parser.add_argument(
        "--hough-thresholds", type=comma_separated_ints, default=[20, 25]
    )
    parser.add_argument(
        "--angle-tolerances",
        type=comma_separated_floats,
        default=[8.0, 12.0],
    )
    parser.add_argument(
        "--cluster-distances",
        type=comma_separated_floats,
        default=[3.0, 5.0, 7.0],
    )
    parser.add_argument(
        "--interval-gaps", type=comma_separated_floats, default=[6.0, 12.0]
    )
    parser.add_argument(
        "--min-support-fractions",
        type=comma_separated_floats,
        default=[0.25, 0.30],
    )
    parser.add_argument(
        "--relative-scores",
        type=comma_separated_floats,
        default=[0.35, 0.45],
    )
    parser.add_argument(
        "--max-edges-values", type=comma_separated_ints, default=[6]
    )
    return parser.parse_args()


def parameter_grid(args: argparse.Namespace) -> list[Parameters]:
    names = (
        "border_fractions",
        "min_line_fractions",
        "max_gap_fractions",
        "hough_thresholds",
        "angle_tolerances",
        "cluster_distances",
        "interval_gaps",
        "min_support_fractions",
        "relative_scores",
        "max_edges_values",
    )
    return [Parameters(*values) for values in itertools.product(*(getattr(args, n) for n in names))]


def validate_args(args: argparse.Namespace, parameters: list[Parameters]) -> None:
    if args.expected_bands < 0:
        raise SystemExit("--expected-bands must be non-negative")
    if args.top < 1:
        raise SystemExit("--top must be at least 1")
    for params in parameters:
        if not 0 <= params.border_fraction < 0.5:
            raise SystemExit("border fractions must be in [0, 0.5)")
        if params.min_line_fraction <= 0 or params.max_gap_fraction < 0:
            raise SystemExit("line and gap fractions must be positive")
        if params.hough_threshold < 1 or params.max_edges < 1:
            raise SystemExit("Hough thresholds and max edge values must be positive")
        if params.angle_tolerance < 0:
            raise SystemExit("angle tolerances must be non-negative")
        if params.cluster_distance < 0 or params.interval_gap < 0:
            raise SystemExit("cluster distances and interval gaps must be non-negative")
        if not 0 <= params.min_support_fraction <= 1:
            raise SystemExit("minimum support fractions must be in [0, 1]")
        if not 0 <= params.relative_score <= 1:
            raise SystemExit("relative scores must be in [0, 1]")


def detector_command(params: Parameters, input_dir: Path, output_dir: Path) -> str:
    detector = Path(__file__).with_name("detect_tactile_bands.py")
    return (
        f"python3 {detector} --input {input_dir} --output-dir {output_dir} "
        f"--border-fraction {params.border_fraction:g} "
        f"--min-line-fraction {params.min_line_fraction:g} "
        f"--max-gap-fraction {params.max_gap_fraction:g} "
        f"--hough-threshold {params.hough_threshold} "
        f"--angle-tolerance {params.angle_tolerance:g} "
        f"--cluster-distance {params.cluster_distance:g} "
        f"--interval-gap {params.interval_gap:g} "
        f"--min-support-fraction {params.min_support_fraction:g} "
        f"--relative-score {params.relative_score:g} "
        f"--max-edges {params.max_edges}"
    )


def main() -> None:
    args = parse_args()
    parameters = parameter_grid(args)
    validate_args(args, parameters)

    inputs = image_paths(args.input_dir)
    if not inputs:
        raise SystemExit(f"No images found in {args.input_dir}")

    output_dir = args.output_dir or args.input_dir / "hyperparameter_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = {}
    for path in inputs:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read {path}")
        images[path] = image

    preprocessed_cache = {}
    segment_cache = {}
    theta_cache = {}
    expected_edges = 2 * args.expected_bands
    summaries = []
    predictions = []

    print(
        f"Evaluating {len(parameters)} parameter combinations on "
        f"{len(inputs)} images (expected bands={args.expected_bands}, "
        f"edges={expected_edges})..."
    )

    for config_number, params in enumerate(parameters, 1):
        config_id = f"C{config_number:04d}"
        counts = []

        for path in inputs:
            preprocess_key = (path, params.border_fraction)
            if preprocess_key not in preprocessed_cache:
                _, edge_map = preprocess(images[path], params.border_fraction)
                preprocessed_cache[preprocess_key] = edge_map

            segment_key = (
                preprocess_key,
                params.min_line_fraction,
                params.max_gap_fraction,
                params.hough_threshold,
            )
            if segment_key not in segment_cache:
                segments = find_segments(
                    preprocessed_cache[preprocess_key],
                    params.min_line_fraction,
                    params.max_gap_fraction,
                    params.hough_threshold,
                )
                segment_cache[segment_key] = segments
                theta_cache[segment_key] = dominant_angle(segments) if segments else None

            segments = segment_cache[segment_key]
            theta = theta_cache[segment_key]
            selected = []
            if theta is not None:
                aligned = [
                    segment
                    for segment in segments
                    if angle_difference(segment.angle, theta)
                    <= math.radians(params.angle_tolerance)
                ]
                candidates = cluster_segments(
                    aligned,
                    theta,
                    params.cluster_distance,
                    params.interval_gap,
                )
                selected = select_edges(
                    candidates,
                    images[path].shape[:2],
                    params.min_support_fraction,
                    params.relative_score,
                    params.max_edges,
                )

            edge_count = len(selected)
            band_count = edge_count / 2
            is_correct = edge_count == expected_edges
            counts.append(edge_count)
            predictions.append(
                {
                    "config_id": config_id,
                    "file": path.name,
                    "edge_count": edge_count,
                    "predicted_band_count": band_count,
                    "expected_band_count": args.expected_bands,
                    "correct": int(is_correct),
                }
            )

        correct_images = sum(count == expected_edges for count in counts)
        absolute_errors = [abs(count / 2 - args.expected_bands) for count in counts]
        summaries.append(
            {
                "config_id": config_id,
                **asdict(params),
                "total_images": len(inputs),
                "correct_images": correct_images,
                "accuracy_percent": 100.0 * correct_images / len(inputs),
                "mean_absolute_band_error": sum(absolute_errors) / len(inputs),
                "zero_edge_images": sum(count == 0 for count in counts),
                "under_count_images": sum(count < expected_edges for count in counts),
                "over_count_images": sum(count > expected_edges for count in counts),
            }
        )

    summaries.sort(
        key=lambda row: (
            -row["accuracy_percent"],
            row["mean_absolute_band_error"],
            row["over_count_images"],
            row["under_count_images"],
        )
    )
    ranks = {row["config_id"]: rank for rank, row in enumerate(summaries, 1)}
    for rank, row in enumerate(summaries, 1):
        row["rank"] = rank
    for row in predictions:
        row["rank"] = ranks[row["config_id"]]
    predictions.sort(key=lambda row: (row["rank"], row["file"]))

    summary_path = output_dir / "ranked_results.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["rank", *[name for name in summaries[0] if name != "rank"]]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    predictions_path = output_dir / "per_image_results.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "rank",
            "config_id",
            "file",
            "edge_count",
            "predicted_band_count",
            "expected_band_count",
            "correct",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    best = summaries[0]
    best_params = Parameters(
        **{field: best[field] for field in Parameters.__dataclass_fields__}
    )
    command = detector_command(
        best_params,
        args.input_dir,
        output_dir / "best_detection_results",
    )
    command_path = output_dir / "best_command.txt"
    command_path.write_text(command + "\n", encoding="utf-8")

    print("\nTop parameter combinations:")
    for row in summaries[: args.top]:
        print(
            f"#{row['rank']:>3} {row['config_id']}: "
            f"{row['correct_images']}/{row['total_images']} correct "
            f"({row['accuracy_percent']:.1f}%), "
            f"max_gap={row['max_gap_fraction']:g}, "
            f"min_line={row['min_line_fraction']:g}, "
            f"hough={row['hough_threshold']}, "
            f"cluster={row['cluster_distance']:g}, "
            f"interval={row['interval_gap']:g}, "
            f"support={row['min_support_fraction']:g}, "
            f"relative={row['relative_score']:g}"
        )
    print(f"\nRanked results: {summary_path}")
    print(f"Per-image results: {predictions_path}")
    print(f"Best detector command: {command_path}")


if __name__ == "__main__":
    main()
