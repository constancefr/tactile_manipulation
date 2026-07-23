#!/usr/bin/env python3
"""Run the best sweep configuration and save annotated/debug images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from detect_tactile_bands import image_paths, process_image


DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1] / "Data" / "Single_rect_Data"
)

PARAMETER_TYPES = {
    "border_fraction": float,
    "min_line_fraction": float,
    "max_gap_fraction": float,
    "hough_threshold": int,
    "angle_tolerance": float,
    "cluster_distance": float,
    "interval_gap": float,
    "min_support_fraction": float,
    "relative_score": float,
    "max_edges": int,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the rank-1 hyperparameter sweep result and save all detector "
            "debug images."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--ranked-results",
        type=Path,
        help="Defaults to <input-dir>/hyperparameter_sweep/ranked_results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <ranked-results-dir>/best_detection_results.",
    )
    return parser.parse_args()


def read_best_row(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(f"Sweep results not found: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit(f"Sweep results are empty: {path}")

    try:
        return min(rows, key=lambda row: int(row["rank"]))
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"Invalid sweep results in {path}: {error}") from error


def detector_args(best_row: dict[str, str]) -> argparse.Namespace:
    values: dict[str, object] = {"save_debug": True}
    for name, converter in PARAMETER_TYPES.items():
        try:
            values[name] = converter(best_row[name])
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"Invalid value for {name}: {error}") from error
    return argparse.Namespace(**values)


def main() -> None:
    args = parse_args()
    ranked_results = args.ranked_results or (
        args.input_dir / "hyperparameter_sweep" / "ranked_results.csv"
    )
    output_dir = args.output_dir or (
        ranked_results.parent / "best_detection_results"
    )

    inputs = image_paths(args.input_dir)
    if not inputs:
        raise SystemExit(f"No images found in {args.input_dir}")

    best_row = read_best_row(ranked_results)
    run_args = detector_args(best_row)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Running rank {best_row['rank']} ({best_row['config_id']}) on "
        f"{len(inputs)} images with debug output enabled..."
    )

    rows = []
    for input_path in inputs:
        output_path = output_dir / f"{input_path.stem}_detected{input_path.suffix}"
        row = process_image(input_path, output_path, run_args)
        rows.append(row)
        debug_path = output_dir / f"{output_path.stem}_debug" / "aligned_segments.png"
        print(
            f"{input_path.name}: edges={row['edge_count']}, "
            f"bands={row['estimated_band_count']}, aligned={debug_path}"
        )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    selected_path = output_dir / "selected_parameters.csv"
    selected_fields = ["rank", "config_id", *PARAMETER_TYPES]
    with selected_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=selected_fields)
        writer.writeheader()
        writer.writerow({field: best_row[field] for field in selected_fields})

    print(f"Annotated and debug images: {output_dir}")
    print(f"Detection summary: {summary_path}")
    print(f"Selected parameters: {selected_path}")


if __name__ == "__main__":
    main()
