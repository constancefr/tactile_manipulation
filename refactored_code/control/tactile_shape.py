#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

"""
Classify the shape and orientation of an object pressed into a DIGIT tactile
sensor's gel, using classical (non-learned) computer vision.

Two segmentation paths, picked automatically based on whether --reference is given:

  A) Background subtraction (used when --reference points to a captured frame
     with nothing touching the sensor -- the preferred, more accurate path).
     absdiff(frame, reference) cancels out the sensor's smooth per-pixel RGB
     lighting gradient directly, leaving just the contact deformation, which
     Otsu-thresholds cleanly since it's no longer competing with a gradient.

  B) Gradient-magnitude fallback (used when no reference frame is available).
     DIGIT frames have a smooth RGB lighting gradient baked in (from the
     sensor's internal LEDs) that swamps simple intensity/color thresholding.
     Contact shows up instead as a comparatively sharp, high-frequency edge
     against that smooth background:
       1. Blur heavily to suppress per-pixel gel-texture/sensor noise.
       2. Sobel gradient magnitude -> the contact boundary stands out as a
          ridge of consistently large gradient, distinct from scattered noise.
       3. Threshold at a percentile of the gradient magnitude (not Otsu --
          Otsu gets fooled into keeping most of the frame because the noise
          floor is itself widespread and low-contrast).

Both paths converge on the same binary-mask -> contour -> convex-hull pipeline:
  - Morphological open (drop noise specks) then close (bridge gaps in the
    contact boundary ring) to get a clean binary blob/ring.
  - Take the convex hull of the largest contour. All target shapes here
    (square/rectangle/triangle/circle/n-gon) are themselves convex, so this
    is a safe simplification that also patches over internal texture holes
    and a ragged boundary without needing more fragile smoothing.
  - Classify via circularity (4*pi*area/perimeter^2) and polygon-approximation
    vertex count; orientation via the object's minimum-area bounding rect.

Usage:
    python3 -m control.tactile_shape --input Data --reference Data/2026-07-21-202339.jpg \\
        --annotate-dir Data/annotated
    python3 -m control.tactile_shape --input Data/2026-07-21-202413.jpg --show
"""

# --- Segmentation tuning: gradient-magnitude fallback (no reference frame) --
BLUR_KERNEL = (15, 15)
BLUR_SIGMA = 3
SOBEL_KSIZE = 5
GRADIENT_PERCENTILE = 92  # threshold = this percentile of gradient magnitude

# --- Segmentation tuning: background subtraction (reference frame given) ---
BGSUB_BLUR_KERNEL = (5, 5)

# --- Segmentation tuning: shared ---------------------------------------------
OPEN_KERNEL_SIZE = 3
CLOSE_KERNEL_SIZE = 15
CLOSE_ITERATIONS = 3
MIN_CONTOUR_AREA = 1500  # px^2; discard smaller contours as noise

# --- Edge-on orientation refinement (Canny + Hough, "edge_contact" cases only) --
# Only applied for edge_contact: a genuinely straight, sharp deformation ridge
# gives Canny a continuous edge to trace. Filled-blob contacts have a soft,
# gradual gel-deformation boundary that Canny fragments into broken pieces
# (verified empirically), so hull/minAreaRect-based orientation stays primary
# for those.
HOUGH_REFINEMENT_ASPECT_MIN = 2.0  # long/short ratio above which Hough refinement is attempted
CANNY_LOW, CANNY_HIGH = 40, 120
HOUGH_THRESHOLD = 30
HOUGH_MIN_LINE_LENGTH = 40
HOUGH_MAX_LINE_GAP = 15
HOUGH_MIN_LINES_FOR_REFINEMENT = 3

# --- Classification tuning ---------------------------------------------------
CIRCLE_CIRCULARITY_MIN = 0.80
SQUARE_ASPECT_TOLERANCE = 0.15  # |w/h - 1| below this -> "square" not "rectangle"
EDGE_ONLY_ASPECT_RATIO = 4.0  # long/short side above this -> treat as edge-on contact
POLY_APPROX_EPSILON_FRAC = 0.06  # fraction of perimeter, for approxPolyDP


@dataclass
class ShapeResult:
    label: str
    orientation_deg: float | None
    vertices: int
    circularity: float
    area: float
    rect_size: tuple[float, float]
    contour: np.ndarray


def _contact_mask_gradient(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Fallback mask when no reference frame is available: segment on gradient
    magnitude, since the sensor's smooth lighting gradient defeats plain thresholding.
    Returns (binary_mask, evidence_image) -- evidence is reused for edge refinement.'''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, BLUR_KERNEL, BLUR_SIGMA)

    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=SOBEL_KSIZE)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=SOBEL_KSIZE)
    mag = cv2.magnitude(gx, gy)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    thresh_value = float(np.percentile(mag_norm, GRADIENT_PERCENTILE))
    _, thresh = cv2.threshold(mag_norm, thresh_value, 255, cv2.THRESH_BINARY)
    return thresh, mag_norm


def _contact_mask_bgsub(img: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Preferred mask when a no-contact reference frame is available: absdiff
    cancels the sensor's lighting gradient directly, so Otsu thresholds cleanly.
    Returns (binary_mask, evidence_image) -- evidence is reused for edge refinement.'''
    img_blur = cv2.GaussianBlur(img.astype(np.float32), BGSUB_BLUR_KERNEL, 0)
    ref_blur = cv2.GaussianBlur(reference.astype(np.float32), BGSUB_BLUR_KERNEL, 0)
    # Correct for frame-to-frame global brightness/exposure drift (e.g. auto-exposure)
    # before diffing -- otherwise a uniform brightness shift alone can make Otsu
    # threshold in a large chunk of background as if it were contact.
    brightness_shift = img_blur.mean(axis=(0, 1)) - ref_blur.mean(axis=(0, 1))
    img_corrected = img_blur - brightness_shift
    diff = cv2.absdiff(img_corrected, ref_blur)
    diff_mag = np.sqrt(np.sum(diff ** 2, axis=2))
    diff_norm = cv2.normalize(diff_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh, diff_norm


def segment_contact(
    img: np.ndarray, reference: np.ndarray | None = None
) -> tuple[np.ndarray | None, np.ndarray]:
    '''Returns (convex_hull_contour_or_None, evidence_image). evidence_image is the
    grayscale signal the mask was thresholded from (diff-vs-reference or gradient
    magnitude), reused by refine_edge_orientation() for Hough line fitting.'''
    if reference is not None:
        mask, evidence = _contact_mask_bgsub(img, reference)
    else:
        mask, evidence = _contact_mask_gradient(img)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=CLOSE_ITERATIONS)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
    if not contours:
        return None, evidence
    largest = max(contours, key=cv2.contourArea)
    return cv2.convexHull(largest), evidence


def refine_edge_orientation(evidence: np.ndarray, hull: np.ndarray) -> float | None:
    '''
    For an elongated (ridge-like) contact, fits a line via Canny + probabilistic
    Hough transform (classical lane-detection technique) restricted to the
    hull's filled interior, and returns the length-weighted dominant angle.
    More precise than minAreaRect for this case -- but NOT used for filled-blob
    shapes, since Canny fragments their soft, gel-compliance-blurred boundary
    into broken pieces rather than continuous lines (verified empirically).

    Uses the filled hull, not just a thin boundary ring: for a ridge-shaped
    contact the sharpest edge is often the bright ridge itself, sitting inside
    the hull's soft outer halo rather than exactly at its silhouette boundary
    (also verified empirically -- a boundary-only ring missed it entirely).
    Returns None if too few line segments are found to trust the fit.
    '''
    band = np.zeros(evidence.shape[:2], dtype=np.uint8)
    cv2.drawContours(band, [hull], -1, 255, thickness=-1)

    edges = cv2.Canny(cv2.GaussianBlur(evidence, (5, 5), 0), CANNY_LOW, CANNY_HIGH)
    edges_masked = cv2.bitwise_and(edges, edges, mask=band)

    lines = cv2.HoughLinesP(
        edges_masked, 1, np.pi / 180, threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH, maxLineGap=HOUGH_MAX_LINE_GAP,
    )
    if lines is None or len(lines) < HOUGH_MIN_LINES_FOR_REFINEMENT:
        return None

    angles, lengths = [], []
    for (x1, y1, x2, y2) in lines[:, 0]:
        lengths.append(float(np.hypot(x2 - x1, y2 - y1)))
        angles.append(float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180))

    # Coarse binning first to find the dominant cluster (handles the wraparound
    # at 0/180 poorly, hence "coarse"), then a length-weighted mean of only the
    # angles inside that bin for a precise, non-quantized estimate -- reporting
    # the bin edge directly would floor every result to a 5-degree grid.
    angles_arr, lengths_arr = np.array(angles), np.array(lengths)
    hist, bin_edges = np.histogram(angles_arr, bins=36, range=(0, 180), weights=lengths_arr)
    dominant_bin = int(np.argmax(hist))
    in_bin = (angles_arr >= bin_edges[dominant_bin]) & (angles_arr < bin_edges[dominant_bin + 1])
    if not np.any(in_bin):
        return float(bin_edges[dominant_bin] + 2.5)
    return float(np.average(angles_arr[in_bin], weights=lengths_arr[in_bin]))


def classify(hull: np.ndarray, evidence: np.ndarray | None = None) -> ShapeResult:
    area = cv2.contourArea(hull)
    peri = cv2.arcLength(hull, True)
    circularity = 4 * math.pi * area / (peri * peri) if peri > 0 else 0.0

    rect = cv2.minAreaRect(hull)
    (_, _), (w, h), angle = rect
    long_side, short_side = max(w, h), max(min(w, h), 1e-6)
    aspect = long_side / short_side

    approx = cv2.approxPolyDP(hull, POLY_APPROX_EPSILON_FRAC * peri, True)
    vertices = len(approx)

    # minAreaRect angle convention: normalize so it refers to the long side's
    # tilt from horizontal, in [0, 180).
    orientation = angle if w >= h else angle + 90
    orientation = orientation % 180

    # Hough line refinement applies to any sufficiently elongated contact
    # (independent of the label threshold below) -- a long thin rectangle and
    # a bare "edge_contact" sliver are the same physical situation for
    # orientation purposes, both benefiting from fitting the actual edge
    # pixels rather than approximating via minAreaRect.
    if evidence is not None and aspect > HOUGH_REFINEMENT_ASPECT_MIN:
        refined = refine_edge_orientation(evidence, hull)
        if refined is not None:
            orientation = refined

    if aspect > EDGE_ONLY_ASPECT_RATIO:
        label = "edge_contact (straight edge -- likely a polygon side seen edge-on)"
    elif circularity >= CIRCLE_CIRCULARITY_MIN and vertices >= 6:
        label = "circle"
        orientation = None  # rotationally symmetric, no meaningful orientation
    elif vertices == 3:
        label = "triangle"
    elif vertices == 4:
        label = "square" if abs(aspect - 1.0) <= SQUARE_ASPECT_TOLERANCE else "rectangle"
    elif vertices == 5:
        label = "pentagon"
    elif vertices == 6:
        label = "hexagon"
    else:
        label = f"unknown ({vertices}-gon, circularity={circularity:.2f})"

    return ShapeResult(
        label=label,
        orientation_deg=orientation,
        vertices=vertices,
        circularity=circularity,
        area=area,
        rect_size=(w, h),
        contour=hull,
    )


def annotate(img: np.ndarray, result: ShapeResult | None) -> np.ndarray:
    vis = img.copy()
    if result is None:
        cv2.putText(vis, "no contact detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis
    cv2.drawContours(vis, [result.contour], -1, (0, 255, 255), 2)
    label = f"{result.label}"
    if result.orientation_deg is not None:
        label += f"  angle={result.orientation_deg:.1f}deg"
        M = cv2.moments(result.contour)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            theta = math.radians(result.orientation_deg)
            length = max(result.rect_size) / 2
            p2 = (int(cx + length * math.cos(theta)), int(cy + length * math.sin(theta)))
            cv2.arrowedLine(vis, (cx, cy), p2, (255, 0, 255), 2, tipLength=0.15)
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vis


def process_image(path: Path, reference: np.ndarray | None = None) -> tuple[np.ndarray, ShapeResult | None]:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"could not read image: {path}")
    hull, evidence = segment_contact(img, reference)
    result = classify(hull, evidence) if hull is not None else None
    return img, result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify shape/orientation from DIGIT tactile images")
    p.add_argument("--input", required=True, help="Image file or directory of images")
    p.add_argument(
        "--reference",
        default=None,
        help="Path to a no-contact reference frame, for background-subtraction segmentation "
        "(recommended -- falls back to gradient-based segmentation if omitted)",
    )
    p.add_argument("--annotate-dir", default=None, help="If set, save annotated debug images here")
    p.add_argument("--show", action="store_true", help="Display each annotated image in a window")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    paths = sorted(input_path.glob("*.jpg")) if input_path.is_dir() else [input_path]

    reference = None
    if args.reference:
        reference_path = Path(args.reference)
        reference = cv2.imread(str(reference_path))
        if reference is None:
            raise SystemExit(f"could not read reference frame: {reference_path}")
        paths = [p for p in paths if p.resolve() != reference_path.resolve()]

    annotate_dir = Path(args.annotate_dir) if args.annotate_dir else None
    if annotate_dir:
        annotate_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        img, result = process_image(path, reference)
        if result is None:
            print(f"{path.name}: no contact detected")
            continue
        angle_str = f"{result.orientation_deg:.1f}deg" if result.orientation_deg is not None else "n/a"
        print(
            f"{path.name}: shape={result.label} orientation={angle_str} "
            f"vertices={result.vertices} circularity={result.circularity:.2f} area={result.area:.0f}"
        )
        if annotate_dir or args.show:
            vis = annotate(img, result)
            if annotate_dir:
                cv2.imwrite(str(annotate_dir / path.name), vis)
            if args.show:
                cv2.imshow(path.name, vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
