"""Small, reusable wrapper around the official DIGIT Python interface.

The wrapper deliberately returns an OpenCV-style BGR ``numpy.ndarray`` so it
can be passed directly to ``TactileBandDetector.detect``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np


class DigitDeviceProtocol(Protocol):
    """Subset of the DIGIT SDK used by :class:`DigitCamera`."""

    STREAMS: dict[str, Any]

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_frame(self) -> np.ndarray: ...

    def set_resolution(self, resolution: Any) -> None: ...

    def set_fps(self, fps: Any) -> None: ...


@dataclass(frozen=True)
class DigitCameraConfig:
    """Connection and stream settings for one DIGIT sensor.

    ``resolution`` and ``fps`` use the names exposed by ``Digit.STREAMS``.
    Leaving them as ``None`` keeps the SDK defaults (normally VGA at 30 fps).
    """

    serial_number: str
    resolution: str | None = None
    fps: str | None = None
    warmup_frames: int = 3

    def __post_init__(self) -> None:
        if not self.serial_number.strip():
            raise ValueError("serial_number cannot be empty")
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames cannot be negative")
        if self.fps is not None and self.resolution is None:
            raise ValueError("fps requires an explicit resolution")


class DigitCamera:
    """Own the connection to a single DIGIT tactile sensor."""

    def __init__(
        self,
        config: DigitCameraConfig,
        *,
        device_factory: Callable[[str], DigitDeviceProtocol] | None = None,
    ) -> None:
        self.config = config
        self._device_factory = device_factory
        self._device: DigitDeviceProtocol | None = None
        self._connected = False
        self._warmed_up = False

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> "DigitCamera":
        if self._connected:
            return self

        factory = self._device_factory or self._load_official_device_factory()
        device = factory(self.config.serial_number)
        device.connect()

        try:
            self._configure_stream(device)
        except Exception:
            device.disconnect()
            raise

        self._device = device
        self._connected = True
        self._warmed_up = False
        return self

    def capture_frame(self) -> np.ndarray:
        """Capture one BGR frame from the connected sensor.

        A few frames are discarded after each new connection so exposure and
        illumination can stabilise before the first image used for inspection.
        """
        device = self._require_device()
        if not self._warmed_up:
            for _ in range(self.config.warmup_frames):
                self._read_valid_frame(device)
            self._warmed_up = True
        return self._read_valid_frame(device).copy()

    def save_frame(self, frame: np.ndarray, path: Path | str) -> Path:
        """Save a captured BGR frame and return its destination path."""
        self._validate_frame(frame)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), frame):
            raise RuntimeError(f"Could not write DIGIT frame: {destination}")
        return destination

    def close(self) -> None:
        if self._device is None:
            self._connected = False
            self._warmed_up = False
            return
        try:
            self._device.disconnect()
        finally:
            self._device = None
            self._connected = False
            self._warmed_up = False

    def __enter__(self) -> "DigitCamera":
        return self.connect()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _configure_stream(self, device: DigitDeviceProtocol) -> None:
        resolution_name = self.config.resolution
        if resolution_name is None:
            return

        streams = getattr(device, "STREAMS", None)
        if not isinstance(streams, dict) or resolution_name not in streams:
            choices = sorted(streams) if isinstance(streams, dict) else []
            raise ValueError(
                f"Unsupported DIGIT resolution {resolution_name!r}; "
                f"available values: {choices}"
            )

        stream = streams[resolution_name]
        device.set_resolution(stream)

        if self.config.fps is None:
            return
        fps_options = stream.get("fps", {}) if isinstance(stream, dict) else {}
        if self.config.fps not in fps_options:
            raise ValueError(
                f"Unsupported FPS {self.config.fps!r} for {resolution_name}; "
                f"available values: {sorted(fps_options)}"
            )
        device.set_fps(fps_options[self.config.fps])

    def _require_device(self) -> DigitDeviceProtocol:
        if not self._connected or self._device is None:
            raise RuntimeError("DIGIT camera is not connected")
        return self._device

    @classmethod
    def _read_valid_frame(cls, device: DigitDeviceProtocol) -> np.ndarray:
        frame = device.get_frame()
        cls._validate_frame(frame)
        return frame

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise RuntimeError("DIGIT SDK did not return a NumPy array")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(
                f"Expected a BGR image with shape (height, width, 3), got {frame.shape}"
            )
        if frame.size == 0:
            raise RuntimeError("DIGIT SDK returned an empty frame")
        if frame.dtype != np.uint8:
            raise RuntimeError(f"Expected uint8 DIGIT frame, got {frame.dtype}")

    @staticmethod
    def _load_official_device_factory() -> Callable[[str], DigitDeviceProtocol]:
        try:
            from digit_interface import Digit
        except ImportError as exc:
            raise RuntimeError(
                "The DIGIT SDK is not installed. Install it with "
                "`pip install digit-interface`."
            ) from exc
        return Digit
