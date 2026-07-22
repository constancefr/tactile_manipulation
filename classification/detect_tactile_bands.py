#!/usr/bin/env python3
"""Detect long band edges in DIGIT tactile images.

For each image, the script saves an annotated copy and reports:
    edge_count
    estimated_band_count = edge_count / 2

It accepts either one image or a directory of images.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class Segment:
    p1: np.ndarray
    p2: np.ndarray
    length: float
    angle: float
    rho: float = 0.0
    t0: float = 0.0
    t1: float = 0.0


@dataclass
class Edge:
    rho: float
    support: float
    score: float


def angle_difference(a: float, b: float) -> float:
    """Difference between unoriented line angles, modulo pi."""
    return abs(((a - b + math.pi / 2) % math.pi) - math.pi / 2)


def preprocess(image: np.ndarray, border_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sigma = max(3.0, 0.06 * min(gray.shape))

    # Remove slow illumination variation from the DIGIT image.
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    enhanced = cv2.normalize(gray - background, None, 0, 255, cv2.NORM_MINMAX)
    enhanced = enhanced.astype(np.uint8)
    enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(enhanced)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    median = float(np.median(enhanced))
    low = int(max(10, 0.66 * median))
    high = int(min(255, max(low + 20, 1.33 * median)))
    edge_map = cv2.Canny(enhanced, low, high, L2gradient=True)
    edge_map = cv2.morphologyEx(
        edge_map,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    # Suppress image/sensor borders, which otherwise look like strong long lines.
    h, w = edge_map.shape
    margin = int(round(border_fraction * min(h, w)))
    if margin > 0:
        edge_map[:margin, :] = 0
        edge_map[-margin:, :] = 0
        edge_map[:, :margin] = 0
        edge_map[:, -margin:] = 0

    return enhanced, edge_map


def find_segments(
    edge_map: np.ndarray,
    min_line_fraction: float,
    max_gap_fraction: float,
    hough_threshold: int,
) -> list[Segment]:
    h, w = edge_map.shape
    scale = min(h, w)
    minimum_length = max(10, int(round(min_line_fraction * scale)))
    maximum_gap = max(2, int(round(max_gap_fraction * scale)))

    lines = cv2.HoughLinesP(
        edge_map,
        rho=1,
        theta=np.pi / 360,
        threshold=hough_threshold,
        minLineLength=minimum_length,
        maxLineGap=maximum_gap,
    )
    if lines is None:
        return []

    segments: list[Segment] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        p1 = np.array([float(x1), float(y1)])
        p2 = np.array([float(x2), float(y2)])
        delta = p2 - p1
        length = float(np.linalg.norm(delta))
        if length < minimum_length:
            continue
        angle = math.atan2(delta[1], delta[0]) % math.pi
        segments.append(Segment(p1, p2, length, angle))
    return segments


def dominant_angle(segments: list[Segment]) -> float:
    """Length-weighted circular histogram over [0, pi)."""
    bins = 180
    histogram = np.zeros(bins, dtype=np.float64)
    for segment in segments:
        index = int(segment.angle / math.pi * bins) % bins
        histogram[index] += segment.length

    # Smooth circularly so lines near 0 and pi are treated as neighbours.
    radius = 4
    padded = np.pad(histogram, (radius, radius), mode="wrap")
    smooth = np.convolve(padded, np.ones(2 * radius + 1), mode="same")
    smooth = smooth[radius:-radius]
    coarse = (int(np.argmax(smooth)) + 0.5) * math.pi / bins

    nearby = [
        s for s in segments if angle_difference(s.angle, coarse) <= math.radians(12)
    ]
    x = sum(s.length * math.cos(2 * s.angle) for s in nearby)
    y = sum(s.length * math.sin(2 * s.angle) for s in nearby)
    return (0.5 * math.atan2(y, x)) % math.pi


def interval_union_length(intervals: list[tuple[float, float]], gap: float) -> float:
    if not intervals:
        return 0.0
    intervals = sorted((min(a, b), max(a, b)) for a, b in intervals)
    start, end = intervals[0]
    total = 0.0
    for next_start, next_end in intervals[1:]:
        if next_start <= end + gap:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def cluster_segments(
    segments: list[Segment],
    theta: float,
    cluster_distance: float,
    interval_gap: float,
) -> list[Edge]:
    tangent = np.array([math.cos(theta), math.sin(theta)])
    normal = np.array([-math.sin(theta), math.cos(theta)])

    for segment in segments:
        midpoint = 0.5 * (segment.p1 + segment.p2)
        segment.rho = float(midpoint @ normal)
        values = sorted((float(segment.p1 @ tangent), float(segment.p2 @ tangent)))
        segment.t0, segment.t1 = values

    ordered = sorted(segments, key=lambda s: s.rho)
    groups: list[list[Segment]] = []
    for segment in ordered:
        if not groups:
            groups.append([segment])
            continue
        group = groups[-1]
        group_rho = float(
            np.average([s.rho for s in group], weights=[s.length for s in group])
        )
        if abs(segment.rho - group_rho) <= cluster_distance:
            group.append(segment)
        else:
            groups.append([segment])

    edges: list[Edge] = []
    for group in groups:
        weights = [s.length for s in group]
        rho = float(np.average([s.rho for s in group], weights=weights))
        support = interval_union_length([(s.t0, s.t1) for s in group], interval_gap)
        score = support + 0.1 * sum(weights)
        edges.append(Edge(rho, support, score))
    return edges


def select_edges(
    candidates: list[Edge],
    image_shape: tuple[int, int],
    minimum_support_fraction: float,
    relative_score: float,
    max_edges: int,
) -> list[Edge]:
    if not candidates:
        return []
    minimum_support = minimum_support_fraction * min(image_shape)
    candidates = [e for e in candidates if e.support >= minimum_support]
    if not candidates:
        return []

    strongest = max(e.score for e in candidates)
    candidates = [e for e in candidates if e.score >= relative_score * strongest]
    candidates = sorted(candidates, key=lambda e: e.score, reverse=True)[:max_edges]
    return sorted(candidates, key=lambda e: e.rho)


def clipped_line(
    rho: float, theta: float, width: int, height: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    tangent = np.array([math.cos(theta), math.sin(theta)])
    normal = np.array([-math.sin(theta), math.cos(theta)])
    point = rho * normal
    extent = 4 * math.hypot(width, height)
    p1 = tuple(np.round(point - extent * tangent).astype(int))
    p2 = tuple(np.round(point + extent * tangent).astype(int))
    ok, p1, p2 = cv2.clipLine((0, 0, width, height), p1, p2)
    return (p1, p2) if ok else None


def annotate(image: np.ndarray, edges: list[Edge], theta: float | None) -> np.ndarray:
    result = image.copy()
    h, w = result.shape[:2]

    if theta is not None:
        for index, edge in enumerate(edges, 1):
            endpoints = clipped_line(edge.rho, theta, w, h)
            if endpoints is None:
                continue
            p1, p2 = endpoints
            # Black outline plus yellow line stays visible on most tactile images.
            cv2.line(result, p1, p2, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.line(result, p1, p2, (0, 255, 255), 3, cv2.LINE_AA)
            midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(
                result,
                f"E{index}",
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                result,
                f"E{index}",
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    edge_count = len(edges)
    bands = edge_count / 2
    label = (
        f"Edges: {edge_count}   Bands: {int(bands)}"
        if edge_count in (2, 4)
        else f"Edges: {edge_count}   Bands: {bands:.1f} (check)"
    )
    cv2.rectangle(result, (6, 6), (390, 44), (0, 0, 0), -1)
    cv2.putText(
        result,
        label,
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def process_image(path: Path, output: Path, args: argparse.Namespace) -> dict[str, object]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read {path}")

    enhanced, edge_map = preprocess(image, args.border_fraction)
    segments = find_segments(
        edge_map,
        args.min_line_fraction,
        args.max_gap_fraction,
        args.hough_threshold,
    )

    theta: float | None = None
    aligned: list[Segment] = []
    selected: list[Edge] = []

    if segments:
        theta = dominant_angle(segments)
        aligned = [
            s
            for s in segments
            if angle_difference(s.angle, theta) <= math.radians(args.angle_tolerance)
        ]
        candidates = cluster_segments(
            aligned,
            theta,
            args.cluster_distance,
            args.interval_gap,
        )
        selected = select_edges(
            candidates,
            image.shape[:2],
            args.min_support_fraction,
            args.relative_score,
            args.max_edges,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), annotate(image, selected, theta))

    if args.save_debug:
        debug_dir = output.parent / f"{output.stem}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "enhanced.png"), enhanced)
        cv2.imwrite(str(debug_dir / "canny_edges.png"), edge_map)

        all_lines = image.copy()
        for s in segments:
            cv2.line(
                all_lines,
                tuple(s.p1.astype(int)),
                tuple(s.p2.astype(int)),
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(debug_dir / "all_hough_segments.png"), all_lines)

        aligned_lines = image.copy()
        for s in aligned:
            cv2.line(
                aligned_lines,
                tuple(s.p1.astype(int)),
                tuple(s.p2.astype(int)),
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(debug_dir / "aligned_segments.png"), aligned_lines)

    edge_count = len(selected)
    return {
        "file": path.name,
        "edge_count": edge_count,
        "estimated_band_count": edge_count / 2,
        "status": "ok" if edge_count in (2, 4) else "check_detection",
        "output": str(output),
    }


def image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Image file or directory")
    parser.add_argument("--output", type=Path, help="Output path for one image")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--save-debug", action="store_true")

    # Tuning parameters. Defaults are intended as starting points, not universal values.
    parser.add_argument("--border-fraction", type=float, default=0.04)
    parser.add_argument("--min-line-fraction", type=float, default=0.22)
    parser.add_argument("--max-gap-fraction", type=float, default=0.06)
    parser.add_argument("--hough-threshold", type=int, default=25)
    parser.add_argument("--angle-tolerance", type=float, default=12.0)
    parser.add_argument("--cluster-distance", type=float, default=7.0)
    parser.add_argument("--interval-gap", type=float, default=12.0)
    parser.add_argument("--min-support-fraction", type=float, default=0.30)
    parser.add_argument("--relative-score", type=float, default=0.45)
    parser.add_argument("--max-edges", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = image_paths(args.input)
    if not inputs:
        raise SystemExit(f"No images found in {args.input}")

    if args.input.is_file():
        outputs = [
            args.output
            or (args.output_dir or args.input.parent)
            / f"{args.input.stem}_detected{args.input.suffix}"
        ]
    else:
        output_dir = args.output_dir or args.input.parent / f"{args.input.name}_results"
        outputs = [output_dir / f"{p.stem}_detected{p.suffix}" for p in inputs]

    rows = []
    for input_path, output_path in zip(inputs, outputs):
        row = process_image(input_path, output_path, args)
        rows.append(row)
        print(
            f"{row['file']}: edges={row['edge_count']}, "
            f"bands={row['estimated_band_count']}, status={row['status']}"
        )

    if args.input.is_dir():
        summary = outputs[0].parent / "summary.csv"
        with summary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
