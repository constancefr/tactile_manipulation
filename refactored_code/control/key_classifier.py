"""Convert tactile detector evidence into the task's binary key label."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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

    def __init__(self, minimum_good_edges: int = 0) -> None:
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
