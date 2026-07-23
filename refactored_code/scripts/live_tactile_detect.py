#!/usr/bin/env python3
"""Run the tactile edge detector on a live DIGIT USB feed instead of static files.

Same image-processing logic as ``scripts/detect_tactile_bands.py``
(``TactileBandDetector`` + ``EmbossedFeatureClassifier``) -- the only thing
that changes is the frame source: ``DigitCamera.capture_frame()`` over USB
instead of ``cv2.imread()`` from disk.

Usage:
    # Reuse a previously captured no-contact reference frame:
    python3 -m scripts.live_tactile_detect --reference Data/Reference_WithoutObject/Withoutanyobject.jpg

    # Or capture a fresh reference at startup (sensor must be untouched):
    python3 -m scripts.live_tactile_detect --capture-reference-frames 5 --save-reference Data/live_reference.jpg
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from control import (
    DetectorConfig,
    EmbossedFeatureClassifier,
    TactileBandDetector,
    draw_classification_overlay,
)
from sensors import DigitCamera, DigitCameraConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--digit-serial",
        help="DIGIT serial number; auto-detected if exactly one sensor is on USB",
    )
    parser.add_argument(
        "--digit-resolution",
        default="VGA",
        help="DIGIT stream name (default VGA/640x480, matching the tuned "
        "detector parameters below -- changing resolution without "
        "re-tuning --diff-gain/--diff-canny-* may change behaviour)",
    )
    parser.add_argument("--digit-fps", default="30fps")
    parser.add_argument(
        "--reference",
        type=Path,
        help="Path to a saved no-contact reference frame (recommended)",
    )
    parser.add_argument(
        "--capture-reference-frames",
        type=int,
        default=0,
        help="If >0 and --reference is not given, capture and average this "
        "many live frames as the reference. The sensor must be untouched "
        "when this runs",
    )
    parser.add_argument(
        "--save-reference",
        type=Path,
        help="Optional path to save a freshly captured reference frame for reuse",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="If set, save each annotated frame and append a CSV log there",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Stop after this many frames (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds to wait between captures",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open a live cv2 window with the annotated feed (needs a display)",
    )
    parser.add_argument("--minimum-good-edges", type=int, default=1)

    # Detector tuning -- same flags/defaults as scripts/detect_tactile_bands.py
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


def resolve_serial(explicit: str | None) -> str:
    if explicit:
        return explicit
    from digit_interface.digit_handler import DigitHandler

    serials = sorted({device["serial"] for device in DigitHandler.list_digits()})
    if not serials:
        raise SystemExit(
            "No DIGIT sensors found on USB. Check the connection or pass --digit-serial."
        )
    if len(serials) > 1:
        raise SystemExit(
            f"Multiple DIGIT sensors found ({serials}); pass --digit-serial to pick one."
        )
    return serials[0]


def capture_reference(camera: DigitCamera, n_frames: int) -> np.ndarray:
    print(
        f"Capturing {n_frames} reference frame(s) -- ensure nothing is "
        "touching the sensor..."
    )
    frames = [camera.capture_frame().astype(np.float32) for _ in range(max(1, n_frames))]
    return (sum(frames) / len(frames)).astype(np.uint8)


def build_detector(args: argparse.Namespace) -> TactileBandDetector:
    return TactileBandDetector(
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


def main() -> None:
    args = parse_args()
    serial = resolve_serial(args.digit_serial)
    digit_config = DigitCameraConfig(
        serial_number=serial,
        resolution=args.digit_resolution,
        fps=args.digit_fps,
    )

    detector = build_detector(args)
    classifier = EmbossedFeatureClassifier(minimum_good_edges=args.minimum_good_edges)

    csv_path = None
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / "live_log.csv"
        if not csv_path.exists():
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(
                    ["timestamp", "frame_index", "edge_count", "angle_deg", "label"]
                )

    with DigitCamera(digit_config) as camera:
        if args.reference is not None:
            reference = cv2.imread(str(args.reference), cv2.IMREAD_COLOR)
            if reference is None:
                raise SystemExit(f"Could not read reference image: {args.reference}")
            probe_shape = camera.capture_frame().shape
            if reference.shape != probe_shape:
                raise SystemExit(
                    f"--reference has shape {reference.shape} but the live feed "
                    f"is {probe_shape} (resolution or sensor mounting/orientation "
                    "differs from when that file was captured -- static "
                    "reference files are not portable across sessions/sensors). "
                    "Capture a fresh one instead with --capture-reference-frames."
                )
        elif args.capture_reference_frames > 0:
            reference = capture_reference(camera, args.capture_reference_frames)
            if args.save_reference:
                camera.save_frame(reference, args.save_reference)
                print(f"Saved captured reference to {args.save_reference}")
        else:
            reference = None
            print(
                "No reference supplied -- falling back to the no-reference "
                "gradient path (less accurate; a saved or captured reference "
                "is strongly recommended)."
            )

        print(f"Connected to DIGIT {serial} ({args.digit_resolution}). "
              "Streaming... (Ctrl+C to stop)")
        frame_index = 0
        try:
            while args.max_frames is None or frame_index < args.max_frames:
                frame = camera.capture_frame()
                result = detector.detect(frame, blank_image=reference)
                classification = classifier.classify(result)
                angle_deg = (
                    math.degrees(result.dominant_angle_rad)
                    if result.dominant_angle_rad is not None
                    else None
                )
                angle_s = f"{angle_deg:.1f}" if angle_deg is not None else "n/a"
                print(
                    f"[{frame_index}] edges={result.edge_count} "
                    f"angle={angle_s} label={classification.label.value}"
                )
                display_image = draw_classification_overlay(
                    result.annotated_image, classification, angle_deg
                )

                if args.output_dir:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(
                        str(args.output_dir / f"{timestamp}_annotated.png"),
                        display_image,
                    )
                    with csv_path.open("a", newline="", encoding="utf-8") as file:
                        csv.writer(file).writerow(
                            [
                                timestamp,
                                frame_index,
                                result.edge_count,
                                angle_deg if angle_deg is not None else "",
                                classification.label.value,
                            ]
                        )

                if args.show:
                    cv2.imshow("DIGIT live tactile detection", display_image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_index += 1
                if args.interval > 0:
                    time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            if args.show:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
