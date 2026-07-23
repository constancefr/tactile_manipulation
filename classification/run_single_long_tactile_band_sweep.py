#!/usr/bin/env python3
"""Sweep long-line detector settings and save the best Single_Long_Data outputs.

This uses the same multi-angle detector path as detect_tactile_bands.py. No
fixed orientation is imposed; each image gets its own dominant line angle.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import shlex
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
    process_image,
    select_edges,
)


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "Data" / "Single_Long_Data"
DEFAULT_OUTPUT = DEFAULT_INPUT / "Best_Outputs"


@dataclass(frozen=True)
class Parameters:
    border_fraction: float
    background_sigma_fraction: float
    clahe_clip_limit: float
    clahe_tile_size: int
    canny_low_multiplier: float
    canny_high_multiplier: float
    morph_kernel_size: int
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


def comma_separated_float_pairs(value: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    try:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            left, right = item.replace("/", ":").split(":", maxsplit=1)
            pairs.append((float(left.strip()), float(right.strip())))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "provide pairs as low:high,low:high"
        ) from error
    if not pairs:
        raise argparse.ArgumentTypeError("provide at least one low:high pair")
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep longer straight-line settings on Single_Long_Data and save "
            "the best annotated/debug outputs."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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
        "--background-sigma-fractions",
        type=comma_separated_floats,
        default=[0.06],
    )
    parser.add_argument(
        "--clahe-clip-limits",
        type=comma_separated_floats,
        default=[2.0],
        help="Lower values amplify background texture less.",
    )
    parser.add_argument("--clahe-tile-sizes", type=comma_separated_ints, default=[8])
    parser.add_argument(
        "--canny-threshold-multipliers",
        type=comma_separated_float_pairs,
        default=[(0.66, 1.33)],
        help="Pairs applied to the enhanced-image median, e.g. 0.66:1.33,1.0:2.0.",
    )
    parser.add_argument("--morph-kernel-sizes", type=comma_separated_ints, default=[3])
    parser.add_argument(
        "--min-line-fractions",
        type=comma_separated_floats,
        default=[0.24, 0.28, 0.30, 0.34],
        help="Longer than the rectangular sweep defaults.",
    )
    parser.add_argument(
        "--max-gap-fractions",
        type=comma_separated_floats,
        default=[0.04, 0.06, 0.08, 0.10, 0.12],
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
        default=[3.0, 5.0],
    )
    parser.add_argument(
        "--interval-gaps", type=comma_separated_floats, default=[6.0, 12.0]
    )
    parser.add_argument(
        "--min-support-fractions",
        type=comma_separated_floats,
        default=[0.30, 0.35, 0.40],
        help="Raised to prefer longer supported edges.",
    )
    parser.add_argument(
        "--relative-scores",
        type=comma_separated_floats,
        default=[0.35, 0.45],
    )
    parser.add_argument("--max-edges-values", type=comma_separated_ints, default=[2])
    return parser.parse_args()


def parameter_grid(args: argparse.Namespace) -> list[Parameters]:
    return [
        Parameters(
            border_fraction,
            background_sigma_fraction,
            clahe_clip_limit,
            clahe_tile_size,
            canny_pair[0],
            canny_pair[1],
            morph_kernel_size,
            min_line_fraction,
            max_gap_fraction,
            hough_threshold,
            angle_tolerance,
            cluster_distance,
            interval_gap,
            min_support_fraction,
            relative_score,
            max_edges,
        )
        for (
            border_fraction,
            background_sigma_fraction,
            clahe_clip_limit,
            clahe_tile_size,
            canny_pair,
            morph_kernel_size,
            min_line_fraction,
            max_gap_fraction,
            hough_threshold,
            angle_tolerance,
            cluster_distance,
            interval_gap,
            min_support_fraction,
            relative_score,
            max_edges,
        ) in itertools.product(
            args.border_fractions,
            args.background_sigma_fractions,
            args.clahe_clip_limits,
            args.clahe_tile_sizes,
            args.canny_threshold_multipliers,
            args.morph_kernel_sizes,
            args.min_line_fractions,
            args.max_gap_fractions,
            args.hough_thresholds,
            args.angle_tolerances,
            args.cluster_distances,
            args.interval_gaps,
            args.min_support_fractions,
            args.relative_scores,
            args.max_edges_values,
        )
    ]


def validate_args(args: argparse.Namespace, parameters: list[Parameters]) -> None:
    if args.expected_bands < 0:
        raise SystemExit("--expected-bands must be non-negative")
    if args.top < 1:
        raise SystemExit("--top must be at least 1")
    for params in parameters:
        if not 0 <= params.border_fraction < 0.5:
            raise SystemExit("border fractions must be in [0, 0.5)")
        if params.background_sigma_fraction <= 0:
            raise SystemExit("background sigma fractions must be positive")
        if params.clahe_clip_limit <= 0:
            raise SystemExit("CLAHE clip limits must be positive")
        if params.clahe_tile_size < 1:
            raise SystemExit("CLAHE tile sizes must be positive")
        if params.canny_low_multiplier <= 0 or params.canny_high_multiplier <= 0:
            raise SystemExit("Canny threshold multipliers must be positive")
        if params.canny_high_multiplier < params.canny_low_multiplier:
            raise SystemExit("Canny high multipliers must be >= low multipliers")
        if params.morph_kernel_size < 1:
            raise SystemExit("morph kernel sizes must be positive")
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


def evaluate(
    args: argparse.Namespace,
    inputs: list[Path],
    parameters: list[Parameters],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
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
        config_id = f"L{config_number:04d}"
        counts = []

        for path in inputs:
            preprocess_key = (
                path,
                params.border_fraction,
                params.background_sigma_fraction,
                params.clahe_clip_limit,
                params.clahe_tile_size,
                params.canny_low_multiplier,
                params.canny_high_multiplier,
                params.morph_kernel_size,
            )
            if preprocess_key not in preprocessed_cache:
                _, edge_map = preprocess(
                    images[path],
                    params.border_fraction,
                    params.background_sigma_fraction,
                    params.clahe_clip_limit,
                    params.clahe_tile_size,
                    params.canny_low_multiplier,
                    params.canny_high_multiplier,
                    params.morph_kernel_size,
                )
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

    return summaries, predictions


def detector_args(params: Parameters) -> argparse.Namespace:
    return argparse.Namespace(save_debug=True, **asdict(params))


def detector_command(params: Parameters, input_dir: Path, output_dir: Path) -> str:
    detector = Path(__file__).with_name("detect_tactile_bands.py")
    pieces = [
        "python3",
        str(detector),
        "--input",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--save-debug",
        "--border-fraction",
        f"{params.border_fraction:g}",
        "--background-sigma-fraction",
        f"{params.background_sigma_fraction:g}",
        "--clahe-clip-limit",
        f"{params.clahe_clip_limit:g}",
        "--clahe-tile-size",
        str(params.clahe_tile_size),
        "--canny-low-multiplier",
        f"{params.canny_low_multiplier:g}",
        "--canny-high-multiplier",
        f"{params.canny_high_multiplier:g}",
        "--morph-kernel-size",
        str(params.morph_kernel_size),
        "--min-line-fraction",
        f"{params.min_line_fraction:g}",
        "--max-gap-fraction",
        f"{params.max_gap_fraction:g}",
        "--hough-threshold",
        str(params.hough_threshold),
        "--angle-tolerance",
        f"{params.angle_tolerance:g}",
        "--cluster-distance",
        f"{params.cluster_distance:g}",
        "--interval-gap",
        f"{params.interval_gap:g}",
        "--min-support-fraction",
        f"{params.min_support_fraction:g}",
        "--relative-score",
        f"{params.relative_score:g}",
        "--max-edges",
        str(params.max_edges),
    ]
    return " ".join(shlex.quote(piece) for piece in pieces)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_best_outputs(
    inputs: list[Path],
    output_dir: Path,
    best: dict[str, object],
) -> None:
    best_params = Parameters(
        **{field: best[field] for field in Parameters.__dataclass_fields__}
    )
    run_args = detector_args(best_params)
    rows = []

    print(
        f"Running rank {best['rank']} ({best['config_id']}) on "
        f"{len(inputs)} images with debug output enabled..."
    )

    for input_path in inputs:
        output_path = output_dir / f"{input_path.stem}_detected{input_path.suffix}"
        row = process_image(input_path, output_path, run_args)
        rows.append(row)
        debug_path = output_dir / f"{output_path.stem}_debug" / "aligned_segments.png"
        print(
            f"{input_path.name}: edges={row['edge_count']}, "
            f"bands={row['estimated_band_count']}, aligned={debug_path}"
        )

    write_csv(output_dir / "summary.csv", rows, list(rows[0].keys()))

    selected_fields = ["rank", "config_id", *Parameters.__dataclass_fields__]
    write_csv(
        output_dir / "selected_parameters.csv",
        [{field: best[field] for field in selected_fields}],
        selected_fields,
    )
    (output_dir / "best_command.txt").write_text(
        detector_command(best_params, inputs[0].parent, output_dir) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    parameters = parameter_grid(args)
    validate_args(args, parameters)

    inputs = image_paths(args.input_dir)
    if not inputs:
        raise SystemExit(f"No images found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries, predictions = evaluate(args, inputs, parameters)

    ranked_path = args.output_dir / "ranked_results.csv"
    write_csv(
        ranked_path,
        summaries,
        ["rank", *[name for name in summaries[0] if name != "rank"]],
    )
    write_csv(
        args.output_dir / "per_image_results.csv",
        predictions,
        [
            "rank",
            "config_id",
            "file",
            "edge_count",
            "predicted_band_count",
            "expected_band_count",
            "correct",
        ],
    )

    best = summaries[0]
    print("\nTop parameter combinations:")
    for row in summaries[: args.top]:
        print(
            f"#{row['rank']:>3} {row['config_id']}: "
            f"{row['correct_images']}/{row['total_images']} correct "
            f"({row['accuracy_percent']:.1f}%), "
            f"clahe={row['clahe_clip_limit']:g}, "
            f"canny={row['canny_low_multiplier']:g}:"
            f"{row['canny_high_multiplier']:g}, "
            f"min_line={row['min_line_fraction']:g}, "
            f"max_gap={row['max_gap_fraction']:g}, "
            f"hough={row['hough_threshold']}, "
            f"angle={row['angle_tolerance']:g}, "
            f"cluster={row['cluster_distance']:g}, "
            f"interval={row['interval_gap']:g}, "
            f"support={row['min_support_fraction']:g}, "
            f"relative={row['relative_score']:g}"
        )

    save_best_outputs(inputs, args.output_dir, best)
    print(f"\nRanked results: {ranked_path}")
    print(f"Best annotated/debug outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
