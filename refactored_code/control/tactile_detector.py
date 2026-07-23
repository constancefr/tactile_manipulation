"""Object-oriented long-edge detector for DIGIT tactile images."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DetectorConfig:
    border_fraction: float = 0.04
    min_line_fraction: float = 0.12
    max_gap_fraction: float = 0.08
    hough_threshold: int = 12
    angle_tolerance_deg: float = 12.0
    cluster_distance_px: float = 7.0
    interval_gap_px: float = 12.0
    min_support_fraction: float = 0.40
    relative_score: float = 0.60
    max_edges: int = 6

    def __post_init__(self) -> None:
        unit_interval_fields = {
            "border_fraction": self.border_fraction,
            "min_line_fraction": self.min_line_fraction,
            "max_gap_fraction": self.max_gap_fraction,
            "min_support_fraction": self.min_support_fraction,
            "relative_score": self.relative_score,
        }
        for name, value in unit_interval_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.hough_threshold <= 0:
            raise ValueError("hough_threshold must be positive")
        if not 0.0 < self.angle_tolerance_deg <= 90.0:
            raise ValueError("angle_tolerance_deg must be in (0, 90]")
        if self.cluster_distance_px < 0.0 or self.interval_gap_px < 0.0:
            raise ValueError("pixel gap parameters cannot be negative")
        if self.max_edges <= 0:
            raise ValueError("max_edges must be positive")


@dataclass(frozen=True)
class Segment:
    p1: np.ndarray
    p2: np.ndarray
    length: float
    angle_rad: float


@dataclass(frozen=True)
class ProjectedSegment:
    segment: Segment
    rho: float
    t0: float
    t1: float


@dataclass(frozen=True)
class DetectedEdge:
    rho: float
    support: float
    score: float


@dataclass(frozen=True)
class DetectionResult:
    """Raw detector output, deliberately separate from good/defect labels."""

    edges: tuple[DetectedEdge, ...]
    dominant_angle_rad: float | None
    segments: tuple[Segment, ...]
    aligned_segments: tuple[Segment, ...]
    source_image: np.ndarray
    enhanced_image: np.ndarray
    edge_map: np.ndarray
    annotated_image: np.ndarray

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def estimated_band_count(self) -> float:
        return self.edge_count / 2.0


@dataclass(frozen=True)
class DetectionRecord:
    source: Path
    annotated_output: Path
    preprocessed_output: Path
    edge_count: int
    estimated_band_count: float

    def as_csv_row(self) -> dict[str, object]:
        return {
            "file": self.source.name,
            "edge_count": self.edge_count,
            "estimated_band_count": self.estimated_band_count,
            "annotated_output": str(self.annotated_output),
            "preprocessed_output": str(self.preprocessed_output),
        }


class TactileBandDetector:
    """Detect long, approximately parallel edges in a DIGIT image."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    def detect(
        self,
        image: np.ndarray,
        *,
        blank_image: np.ndarray | None = None,
    ) -> DetectionResult:
        """Run detection on an in-memory BGR image."""
        self._validate_image(image, "image")
        if blank_image is not None:
            self._validate_image(blank_image, "blank_image")
            if blank_image.shape[:2] != image.shape[:2]:
                raise ValueError("blank_image dimensions must match image dimensions")

        normal = self._detect_variant(image, blank_image=None)
        if blank_image is None:
            return normal

        subtracted = self._detect_variant(image, blank_image=blank_image)
        # Choose using detector evidence only. Good/defect semantics belong in a
        # later classifier and are intentionally not encoded here.
        return max(
            (normal, subtracted),
            key=lambda result: sum(edge.score for edge in result.edges),
        )

    def detect_file(
        self,
        path: Path | str,
        *,
        blank_path: Path | str | None = None,
    ) -> DetectionResult:
        image_path = Path(path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        blank_image = None
        if blank_path is not None:
            blank_image = cv2.imread(str(Path(blank_path)), cv2.IMREAD_COLOR)
            if blank_image is None:
                raise RuntimeError(f"Could not read blank image: {blank_path}")
        return self.detect(image, blank_image=blank_image)

    def save_result(
        self,
        result: DetectionResult,
        source: Path | str,
        output_dir: Path | str,
        *,
        save_debug: bool = False,
    ) -> DetectionRecord:
        source_path = Path(source)
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        suffix = source_path.suffix if source_path.suffix else ".png"
        annotated_path = destination / f"{source_path.stem}_detected{suffix}"
        preprocessed_path = destination / f"{source_path.stem}_preprocessed.png"
        self._write_image(annotated_path, result.annotated_image)
        self._write_image(preprocessed_path, result.enhanced_image)

        if save_debug:
            debug_dir = destination / f"{source_path.stem}_debug"
            self._save_debug_images(result, debug_dir)

        return DetectionRecord(
            source=source_path,
            annotated_output=annotated_path,
            preprocessed_output=preprocessed_path,
            edge_count=result.edge_count,
            estimated_band_count=result.estimated_band_count,
        )

    def process_path(
        self,
        input_path: Path | str,
        *,
        output_dir: Path | str | None = None,
        blank_path: Path | str | None = None,
        save_debug: bool = False,
    ) -> list[DetectionRecord]:
        source = Path(input_path)
        files = list(self.image_paths(source))
        if not files:
            raise ValueError(f"No supported images found in {source}")

        if output_dir is None:
            result_name = (
                f"{source.stem}_results" if source.is_file() else f"{source.name}_results"
            )
            destination = source.parent / result_name
        else:
            destination = Path(output_dir)

        records = [
            self.save_result(
                self.detect_file(path, blank_path=blank_path),
                path,
                destination,
                save_debug=save_debug,
            )
            for path in files
        ]
        if source.is_dir():
            self.write_summary(records, destination / "summary.csv")
        return records

    @staticmethod
    def image_paths(path: Path | str) -> Iterable[Path]:
        source = Path(path)
        if source.is_file():
            if source.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image extension: {source.suffix}")
            yield source
            return
        if source.is_dir():
            yield from sorted(
                item
                for item in source.iterdir()
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            )
            return
        raise FileNotFoundError(source)

    @staticmethod
    def write_summary(
        records: Iterable[DetectionRecord],
        path: Path | str,
    ) -> None:
        rows = [record.as_csv_row() for record in records]
        if not rows:
            return
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _detect_variant(
        self,
        image: np.ndarray,
        *,
        blank_image: np.ndarray | None,
    ) -> DetectionResult:
        enhanced, edge_map = self._preprocess(image, blank_image)
        segments = self._find_segments(edge_map)
        if not segments:
            annotated = self._annotate(image, (), None)
            return DetectionResult(
                edges=(),
                dominant_angle_rad=None,
                segments=(),
                aligned_segments=(),
                source_image=image.copy(),
                enhanced_image=enhanced,
                edge_map=edge_map,
                annotated_image=annotated,
            )

        theta = self._dominant_angle(segments)
        tolerance = math.radians(self.config.angle_tolerance_deg)
        aligned = tuple(
            segment
            for segment in segments
            if self._angle_difference(segment.angle_rad, theta) <= tolerance
        )
        candidates = self._cluster_segments(aligned, theta)
        selected = self._select_edges(candidates, image.shape[:2])
        annotated = self._annotate(image, selected, theta)
        return DetectionResult(
            edges=selected,
            dominant_angle_rad=theta,
            segments=segments,
            aligned_segments=aligned,
            source_image=image.copy(),
            enhanced_image=enhanced,
            edge_map=edge_map,
            annotated_image=annotated,
        )

    def _preprocess(
        self,
        image: np.ndarray,
        blank_image: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if blank_image is not None:
            blank_gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY).astype(
                np.float32
            )
            enhanced_float = gray - blank_gray
        else:
            sigma = max(3.0, 0.06 * min(gray.shape))
            background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
            enhanced_float = gray - background

        enhanced = cv2.normalize(
            enhanced_float, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
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

        height, width = edge_map.shape
        margin = int(round(self.config.border_fraction * min(height, width)))
        if margin > 0:
            edge_map[:margin, :] = 0
            edge_map[-margin:, :] = 0
            edge_map[:, :margin] = 0
            edge_map[:, -margin:] = 0
        return enhanced, edge_map

    def _find_segments(self, edge_map: np.ndarray) -> tuple[Segment, ...]:
        height, width = edge_map.shape
        scale = min(height, width)
        minimum_length = max(
            10, int(round(self.config.min_line_fraction * scale))
        )
        maximum_gap = max(
            2, int(round(self.config.max_gap_fraction * scale))
        )
        lines = cv2.HoughLinesP(
            edge_map,
            rho=1,
            theta=np.pi / 360,
            threshold=self.config.hough_threshold,
            minLineLength=minimum_length,
            maxLineGap=maximum_gap,
        )
        if lines is None:
            return ()

        segments: list[Segment] = []
        for x1, y1, x2, y2 in lines[:, 0]:
            p1 = np.array([float(x1), float(y1)])
            p2 = np.array([float(x2), float(y2)])
            delta = p2 - p1
            length = float(np.linalg.norm(delta))
            if length >= minimum_length:
                segments.append(
                    Segment(
                        p1=p1,
                        p2=p2,
                        length=length,
                        angle_rad=math.atan2(delta[1], delta[0]) % math.pi,
                    )
                )
        return tuple(segments)

    def _dominant_angle(self, segments: tuple[Segment, ...]) -> float:
        bins = 180
        histogram = np.zeros(bins, dtype=np.float64)
        for segment in segments:
            index = int(segment.angle_rad / math.pi * bins) % bins
            histogram[index] += segment.length

        radius = 4
        padded = np.pad(histogram, (radius, radius), mode="wrap")
        smooth = np.convolve(
            padded,
            np.ones(2 * radius + 1),
            mode="same",
        )[radius:-radius]
        coarse = (int(np.argmax(smooth)) + 0.5) * math.pi / bins
        nearby = tuple(
            segment
            for segment in segments
            if self._angle_difference(segment.angle_rad, coarse)
            <= math.radians(12.0)
        )
        x = sum(
            segment.length * math.cos(2.0 * segment.angle_rad)
            for segment in nearby
        )
        y = sum(
            segment.length * math.sin(2.0 * segment.angle_rad)
            for segment in nearby
        )
        return (0.5 * math.atan2(y, x)) % math.pi

    def _cluster_segments(
        self,
        segments: tuple[Segment, ...],
        theta: float,
    ) -> tuple[DetectedEdge, ...]:
        tangent = np.array([math.cos(theta), math.sin(theta)])
        normal = np.array([-math.sin(theta), math.cos(theta)])
        projected: list[ProjectedSegment] = []
        for segment in segments:
            midpoint = 0.5 * (segment.p1 + segment.p2)
            endpoints = sorted(
                (float(segment.p1 @ tangent), float(segment.p2 @ tangent))
            )
            projected.append(
                ProjectedSegment(
                    segment=segment,
                    rho=float(midpoint @ normal),
                    t0=endpoints[0],
                    t1=endpoints[1],
                )
            )

        groups: list[list[ProjectedSegment]] = []
        for segment in sorted(projected, key=lambda item: item.rho):
            if not groups:
                groups.append([segment])
                continue
            group = groups[-1]
            group_rho = float(
                np.average(
                    [item.rho for item in group],
                    weights=[item.segment.length for item in group],
                )
            )
            if abs(segment.rho - group_rho) <= self.config.cluster_distance_px:
                group.append(segment)
            else:
                groups.append([segment])

        edges: list[DetectedEdge] = []
        for group in groups:
            weights = [item.segment.length for item in group]
            rho = float(
                np.average([item.rho for item in group], weights=weights)
            )
            support = self._interval_union_length(
                [(item.t0, item.t1) for item in group],
                self.config.interval_gap_px,
            )
            edges.append(
                DetectedEdge(
                    rho=rho,
                    support=support,
                    score=support + 0.1 * sum(weights),
                )
            )
        return tuple(edges)

    def _select_edges(
        self,
        candidates: tuple[DetectedEdge, ...],
        image_shape: tuple[int, int],
    ) -> tuple[DetectedEdge, ...]:
        if not candidates:
            return ()
        minimum_support = self.config.min_support_fraction * min(image_shape)
        supported = tuple(
            edge for edge in candidates if edge.support >= minimum_support
        )
        if not supported:
            return ()
        strongest_score = max(edge.score for edge in supported)
        selected = [
            edge
            for edge in supported
            if edge.score >= self.config.relative_score * strongest_score
        ]
        selected.sort(key=lambda edge: edge.score, reverse=True)
        selected = selected[: self.config.max_edges]
        selected.sort(key=lambda edge: edge.rho)
        return tuple(selected)

    def _annotate(
        self,
        image: np.ndarray,
        edges: tuple[DetectedEdge, ...],
        theta: float | None,
    ) -> np.ndarray:
        result = image.copy()
        height, width = result.shape[:2]
        if theta is not None:
            for index, edge in enumerate(edges, 1):
                endpoints = self._clipped_line(edge.rho, theta, width, height)
                if endpoints is None:
                    continue
                p1, p2 = endpoints
                cv2.line(result, p1, p2, (0, 0, 0), 7, cv2.LINE_AA)
                cv2.line(result, p1, p2, (0, 255, 255), 3, cv2.LINE_AA)
                midpoint = (
                    (p1[0] + p2[0]) // 2,
                    (p1[1] + p2[1]) // 2,
                )
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

        label = (
            f"Edges: {len(edges)}   "
            f"Estimated bands: {len(edges) / 2.0:.1f}"
        )
        cv2.rectangle(result, (6, 6), (430, 44), (0, 0, 0), -1)
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

    def _save_debug_images(
        self,
        result: DetectionResult,
        debug_dir: Path,
    ) -> None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        self._write_image(debug_dir / "enhanced.png", result.enhanced_image)
        self._write_image(debug_dir / "canny_edges.png", result.edge_map)

        all_lines = result.source_image.copy()
        for segment in result.segments:
            cv2.line(
                all_lines,
                tuple(segment.p1.astype(int)),
                tuple(segment.p2.astype(int)),
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
        self._write_image(debug_dir / "all_hough_segments.png", all_lines)

        aligned_lines = result.source_image.copy()
        for segment in result.aligned_segments:
            cv2.line(
                aligned_lines,
                tuple(segment.p1.astype(int)),
                tuple(segment.p2.astype(int)),
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        self._write_image(debug_dir / "aligned_segments.png", aligned_lines)

    @staticmethod
    def _angle_difference(first: float, second: float) -> float:
        return abs(
            ((first - second + math.pi / 2.0) % math.pi) - math.pi / 2.0
        )

    @staticmethod
    def _interval_union_length(
        intervals: list[tuple[float, float]],
        gap: float,
    ) -> float:
        if not intervals:
            return 0.0
        ordered = sorted((min(a, b), max(a, b)) for a, b in intervals)
        start, end = ordered[0]
        total = 0.0
        for next_start, next_end in ordered[1:]:
            if next_start <= end + gap:
                end = max(end, next_end)
            else:
                total += end - start
                start, end = next_start, next_end
        return total + end - start

    @staticmethod
    def _clipped_line(
        rho: float,
        theta: float,
        width: int,
        height: int,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        tangent = np.array([math.cos(theta), math.sin(theta)])
        normal = np.array([-math.sin(theta), math.cos(theta)])
        point = rho * normal
        extent = 4.0 * math.hypot(width, height)
        p1 = tuple(np.round(point - extent * tangent).astype(int))
        p2 = tuple(np.round(point + extent * tangent).astype(int))
        ok, clipped_p1, clipped_p2 = cv2.clipLine(
            (0, 0, width, height), p1, p2
        )
        return (clipped_p1, clipped_p2) if ok else None

    @staticmethod
    def _validate_image(image: np.ndarray, name: str) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{name} must be a BGR image with shape HxWx3")
        if image.size == 0:
            raise ValueError(f"{name} cannot be empty")

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write image: {path}")
