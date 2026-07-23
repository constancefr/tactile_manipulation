#!/usr/bin/env python3
"""Detect long embossed-feature edges in one DIGIT image or a directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from control import DetectorConfig, TactileBandDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--blank-image", type=Path)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--border-fraction", type=float, default=0.04)
    parser.add_argument("--min-line-fraction", type=float, default=0.12)
    parser.add_argument("--max-gap-fraction", type=float, default=0.08)
    parser.add_argument("--hough-threshold", type=int, default=12)
    parser.add_argument("--angle-tolerance", type=float, default=12.0)
    parser.add_argument("--cluster-distance", type=float, default=7.0)
    parser.add_argument("--interval-gap", type=float, default=12.0)
    parser.add_argument("--min-support-fraction", type=float, default=0.40)
    parser.add_argument("--relative-score", type=float, default=0.60)
    parser.add_argument("--max-edges", type=int, default=6)
    parser.add_argument("--diff-blur-kernel", type=int, default=5)
    parser.add_argument("--diff-gain", type=float, default=12.0)
    parser.add_argument("--diff-canny-low", type=int, default=40)
    parser.add_argument("--diff-canny-high", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = TactileBandDetector(
        DetectorConfig(
            border_fraction=args.border_fraction,
            min_line_fraction=args.min_line_fraction,
            max_gap_fraction=args.max_gap_fraction,
            hough_threshold=args.hough_threshold,
            angle_tolerance_deg=args.angle_tolerance,
            cluster_distance_px=args.cluster_distance,
            interval_gap_px=args.interval_gap,
            min_support_fraction=args.min_support_fraction,
            relative_score=args.relative_score,
            max_edges=args.max_edges,
            diff_blur_kernel=args.diff_blur_kernel,
            diff_gain=args.diff_gain,
            diff_canny_low=args.diff_canny_low,
            diff_canny_high=args.diff_canny_high,
        )
    )
    records = detector.process_path(
        args.input,
        output_dir=args.output_dir,
        blank_path=args.blank_image,
        save_debug=args.save_debug,
    )
    for record in records:
        print(
            f"{record.source.name}: edges={record.edge_count}, "
            f"estimated_bands={record.estimated_band_count:.1f}, "
            f"output={record.annotated_output}"
        )


if __name__ == "__main__":
    main()
