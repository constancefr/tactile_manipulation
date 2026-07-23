"""Convert tactile detector evidence into the task's binary key label."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .tactile_detector import DetectionResult


class KeyLabel(str, Enum):
    GOOD = "good"
    DEFECT = "defect"


@dataclass(frozen=True)
class KeyClassification:
    label: KeyLabel
    edge_count: int
    minimum_good_edges: int


class EmbossedFeatureClassifier:
    """Temporary edge-count rule for flat versus embossed keys.

    The intended evidence is approximately zero long edges for a flat key and
    two long edges for one embossed line. This simple rule is intentionally
    isolated so it can later be replaced without changing the robot sequence.
    """

    def __init__(self, minimum_good_edges: int = 1) -> None:
        if minimum_good_edges < 0:
            raise ValueError("minimum_good_edges must be zero or positive")
        self.minimum_good_edges = int(minimum_good_edges)

    def classify(self, detection: DetectionResult) -> KeyClassification:
        edge_count = detection.edge_count
        label = (
            KeyLabel.GOOD
            if edge_count >= self.minimum_good_edges
            else KeyLabel.DEFECT
        )
        return KeyClassification(
            label=label,
            edge_count=edge_count,
            minimum_good_edges=self.minimum_good_edges,
        )

    def dummy_classify(self, detection: DetectionResult) -> KeyClassification:
        """Dummy classifier that always returns GOOD."""
        return KeyClassification(
            label=KeyLabel.GOOD,
            edge_count=detection.edge_count,
            minimum_good_edges=self.minimum_good_edges,
        )


def draw_classification_overlay(
    image: np.ndarray,
    classification: KeyClassification,
    angle_deg: float | None,
) -> np.ndarray:
    """Add a classification-label + angle banner to a detector-annotated image.

    ``TactileBandDetector._annotate`` only draws the raw edge count -- it has
    no knowledge of the classifier, which runs one layer up on its output.
    This keeps that separation but gives any saved/displayed frame the full
    picture: label and angle, not just edges. Shared by the live-feed script
    and ``KeySortingTask`` so both save/show the same overlay.
    """
    out = image.copy()
    angle_text = f"{angle_deg:.1f} deg" if angle_deg is not None else "n/a"
    label_text = f"{classification.label.value.upper()}   angle: {angle_text}"
    color = (0, 255, 0) if classification.label is KeyLabel.GOOD else (0, 0, 255)

    height = out.shape[0]
    cv2.rectangle(out, (6, height - 44), (430, height - 6), (0, 0, 0), -1)
    cv2.putText(
        out,
        label_text,
        (14, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return out
