#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import control.tactile_shape as ts

"""
Step-by-step visualization of tactile_shape.py's existing no-reference pipeline
for a single image: raw frame -> gradient magnitude -> threshold -> morphology
-> convex hull -> Canny edges -> Hough line segments -> classification.
Produces one annotated 9-panel image per input so each stage can be inspected
directly, rather than just trusting the final printed label. The Canny/Hough
panels only show real content when aspect > HOUGH_REFINEMENT_ASPECT_MIN
(mirrors classify() exactly); otherwise they're blank with a note, since that
refinement wasn't applied.

Usage:
    python3 -m control.tactile_shape_debug --input Data \\
        --filter rectangle --out-dir Data/debug_steps
"""

PANEL_LABEL_COLOR = (0, 255, 0)
PANEL_W, PANEL_H = 320, 240  # each stage thumbnail size in the grid


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PANEL_LABEL_COLOR, 1)
    return out


def to_bgr(gray_or_bgr: np.ndarray) -> np.ndarray:
    if gray_or_bgr.ndim == 2:
        return cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2BGR)
    return gray_or_bgr


def resize(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (PANEL_W, PANEL_H))


def build_debug_panel(
    path: Path,
) -> tuple[np.ndarray, ts.ShapeResult | None, dict]:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"could not read image: {path}")

    # Use tactile_shape.py's existing no-reference segmentation path.
    thresh, evidence = ts._contact_mask_gradient(img)

    # --- Stage 3: morphological open (remove speckle noise) ---
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ts.OPEN_KERNEL_SIZE, ts.OPEN_KERNEL_SIZE))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel, iterations=1)

    # --- Stage 4: morphological close (bridge gaps into a solid blob/ring) ---
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ts.CLOSE_KERNEL_SIZE, ts.CLOSE_KERNEL_SIZE))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=ts.CLOSE_ITERATIONS)

    # --- Stage 5: largest contour + convex hull ---
    hull, _ = ts.segment_contact(img)
    contour_vis = img.copy()
    for idx, hull in enumerate(hulls, start=1):
        color = ts.OBJECT_COLORS[(idx - 1) % len(ts.OBJECT_COLORS)]
        cv2.drawContours(contour_vis, [hull], -1, color, 2)

    # --- Stage 6: feature extraction + classification ---
    result = ts.classify(hull, evidence) if hull is not None else None
    features = {}
    final_vis = img.copy()
    aspect = 0.0
    if result is not None:
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect).astype(int)
        approx = cv2.approxPolyDP(hull, ts.POLY_APPROX_EPSILON_FRAC * cv2.arcLength(hull, True), True)
        aspect = max(result.rect_size) / max(min(result.rect_size), 1e-6)
        features.append({
            "area": result.area,
            "perimeter": cv2.arcLength(hull, True),
            "circularity": result.circularity,
            "vertices": result.vertices,
            "rect_w": result.rect_size[0],
            "rect_h": result.rect_size[1],
            "aspect": aspect,
            "raw_minAreaRect_angle": rect[2],
            "gradient_percentile": ts.GRADIENT_PERCENTILE,
        }
        final_vis = ts.annotate(img, result)
        cv2.drawContours(final_vis, [box], -1, (255, 0, 255), 1)
        for pt in approx:
            cv2.circle(final_vis, tuple(pt[0]), 4, (0, 0, 255), -1)

    # --- Stage 7/8: Canny edges + Hough line segments, per object that
    # qualifies (aspect > HOUGH_REFINEMENT_ASPECT_MIN -- mirrors classify()'s
    # own gating exactly), overlaid together in each object's color ---
    canny_vis = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
    hough_vis = img.copy()
    hough_note = "not applicable (aspect <= threshold)"
    n_lines = 0
    if hull is not None and aspect > ts.HOUGH_REFINEMENT_ASPECT_MIN:
        filled = np.zeros(evidence.shape[:2], dtype=np.uint8)
        cv2.drawContours(filled, [hull], -1, 255, thickness=-1)
        canny_vis = cv2.Canny(cv2.GaussianBlur(evidence, (5, 5), 0), ts.CANNY_LOW, ts.CANNY_HIGH)
        canny_vis = cv2.bitwise_and(canny_vis, canny_vis, mask=filled)

        lines = cv2.HoughLinesP(
            edges_masked, 1, np.pi / 180, threshold=ts.HOUGH_THRESHOLD,
            minLineLength=ts.HOUGH_MIN_LINE_LENGTH, maxLineGap=ts.HOUGH_MAX_LINE_GAP,
        )
        n_lines = 0 if lines is None else len(lines)
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        refined = result.orientation_deg if result is not None else None
        hough_note = f"{n_lines} segments -> refined angle={refined:.1f}deg" if refined is not None else f"{n_lines} segments (too few to refine)"

    if result is not None:
        # Expose the existing Hough output so an adapter can apply task-level
        # GOOD/DEFECT semantics without running a second line detector.
        features["num_edges"] = n_lines

    stages = [
        (resize(img), "1. raw frame"),
        (resize(to_bgr(evidence)), "2. gradient magnitude"),
        (
            resize(to_bgr(thresh)),
            f"3. gradient threshold (p={ts.GRADIENT_PERCENTILE})",
        ),
        (resize(to_bgr(opened)), "4. morph open (denoise)"),
        (resize(to_bgr(closed)), "5. morph close (fill ring)"),
        (resize(contour_vis), f"6. {len(hulls)} object(s) -> convex hulls"),
        (resize(canny_vis), "7. Canny edges (filled-hull mask, per object)"),
        (resize(hough_vis), f"8. Hough: {hough_summary}"),
        (resize(final_vis), "9. classify + orientation"),
    ]
    labeled = [label(im, txt) for im, txt in stages]
    row1 = np.hstack(labeled[0:3])
    row2 = np.hstack(labeled[3:6])
    row3 = np.hstack(labeled[6:9])
    panel = np.vstack([row1, row2, row3])
    return panel, results, features


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Step-by-step visualization of tactile_shape.py")
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--filter", default=None, help="Only process images where at least one detected object's label contains this substring")
    args = p.parse_args()

    input_path = Path(args.input)
    paths = sorted(input_path.glob("*.jpg")) if input_path.is_dir() else [input_path]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        panel, result, features = build_debug_panel(path)
        if result is None:
            continue
        if args.filter and not any(args.filter in r.label for r in results):
            continue
        out_path = out_dir / path.name
        cv2.imwrite(str(out_path), panel)
        summary = "; ".join(
            f"#{i} label={r.label} orientation={r.orientation_deg} " + " ".join(f"{k}={v:.2f}" for k, v in f.items())
            for i, (r, f) in enumerate(zip(results, features), start=1)
        )
        rect_count = sum(1 for r in results if r.label in ("rectangle", "square"))
        print(f"{path.name}: {len(results)} object(s), {rect_count} rectangle-like -- {summary}")


if __name__ == "__main__":
    main()
