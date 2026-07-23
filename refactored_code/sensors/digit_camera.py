"""Small, reusable wrapper around the official DIGIT Python interface.

The wrapper deliberately returns an OpenCV-style BGR ``numpy.ndarray`` so it
can be passed directly to ``TactileBandDetector.detect``.
"""

from __future__ import annotations

import time
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

    def get_frame(self, transpose: bool = False) -> np.ndarray: ...

    def set_resolution(self, resolution: Any) -> None: ...

    def set_fps(self, fps: Any) -> None: ...

    def set_intensity(self, intensity: int) -> int: ...


@dataclass(frozen=True)
class DigitCameraConfig:
    """Connection and stream settings for one DIGIT sensor.

    ``resolution`` and ``fps`` use the names exposed by ``Digit.STREAMS``.
    Leaving ``resolution`` as ``None`` keeps the SDK's own connect-time
    default, which is QVGA (320x240) at 60 fps, NOT VGA -- verified against
    the ``digit_interface.Digit.connect()`` source. QVGA is a quarter the
    frame area of the 640x480 VGA stream every detector parameter and
    reference image in this project was tuned against, and UVC low-res modes
    are typically a hardware crop rather than a clean downscale, so an
    unset resolution can look like a cropped frame. Pass ``resolution="VGA"``
    explicitly unless you have a specific reason not to.
    """

    serial_number: str
    resolution: str | None = None
    fps: str | None = None
    warmup_frames: int = 3
    led_intensity: int | None = 15
    stream_settle_delay_sec: float = 8.0
    connect_retries: int = 3
    flush_frames_per_capture: int = 4

    def __post_init__(self) -> None:
        if not self.serial_number.strip():
            raise ValueError("serial_number cannot be empty")
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames cannot be negative")
        if self.fps is not None and self.resolution is None:
            raise ValueError("fps requires an explicit resolution")
        if self.led_intensity is not None and not 0 <= self.led_intensity <= 15:
            raise ValueError("led_intensity must be in [0, 15]")
        if self.stream_settle_delay_sec < 0.0:
            raise ValueError("stream_settle_delay_sec cannot be negative")
        if self.connect_retries < 1:
            raise ValueError("connect_retries must be at least 1")
        if self.flush_frames_per_capture < 0:
            raise ValueError("flush_frames_per_capture cannot be negative")


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
        last_error: Exception | None = None
        for attempt in range(1, self.config.connect_retries + 1):
            # A fresh factory() call re-runs device discovery from scratch.
            # This matters because of a confirmed quirk on at least this
            # hardware: the DIGIT's composite USB device registers two
            # /dev/videoN nodes under the *same* serial number (verified: one
            # genuinely streams, the other times out on every read), and
            # which one digit_interface's udev-based lookup returns first is
            # not stable across reconnects -- it changed between consecutive
            # attempts in testing with no physical change at all. Retrying
            # the whole discovery-and-connect sequence, not just the read,
            # gives a real chance of landing on the working node.
            device = factory(self.config.serial_number)
            try:
                device.connect()
                self._configure_stream(device)
                self._settle_stream(device)
            except Exception as exc:
                last_error = exc
                device.disconnect()
                continue

            self._device = device
            self._connected = True
            # Only skip capture_frame()'s own warmup if _settle_stream
            # actually read live frames (stream_settle_delay_sec > 0);
            # with it disabled, no frames were consumed yet.
            self._warmed_up = bool(self.config.stream_settle_delay_sec)
            return self

        raise RuntimeError(
            "Could not establish a working DIGIT connection after "
            f"{self.config.connect_retries} attempt(s): {last_error}"
        ) from last_error

    def _settle_stream(self, device: DigitDeviceProtocol) -> None:
        """Actively read and discard frames for stream_settle_delay_sec.

        The sensor's own auto-exposure/colour convergence takes several
        seconds after the stream starts and only progresses while frames are
        actually being pulled -- measured on real hardware: a blind
        time.sleep() of the same duration with no reads during it left
        colour balance unchanged (heavily green-dominant, blue near zero),
        while continuously reading during the wait converged to a correctly
        balanced image by ~5s and held steady after. Confirmed independently
        in `cheese`: it shows the same initial green frame before "quickly"
        (a few seconds) settling -- it is also continuously pulling frames.
        This is not caused by (and not fixed by) resolution/fps switching or
        the LED-intensity call in _configure_stream, both of which complete
        almost instantly. A read failure here (e.g. the wrong-node issue
        above) propagates out so connect() can retry with fresh discovery.
        """
        if not self.config.stream_settle_delay_sec:
            return
        deadline = time.monotonic() + self.config.stream_settle_delay_sec
        while time.monotonic() < deadline:
            device.get_frame(transpose=True)

    def capture_frame(self) -> np.ndarray:
        """Capture one BGR frame reflecting the sensor's current state.

        A few frames are discarded after each new connection so exposure and
        illumination can stabilise before the first image used for
        inspection. Separately, and on *every* call (not just the first):
        the V4L2/UVC layer buffers several frames internally, so a single
        .read() after any gap since the last capture_frame() call (gripper
        motion time, a sleep between two inspection frames, etc.) can return
        a stale, already-buffered frame rather than the sensor's current
        state -- confirmed to matter in practice, since a reference frame
        captured before the gripper closes and an inspection frame captured
        shortly after otherwise risked both coming from the same pre-close
        buffered backlog. flush_frames_per_capture drains that backlog
        before every real read, independent of the one-time post-connect
        warmup above.
        """
        device = self._require_device()
        if not self._warmed_up:
            for _ in range(self.config.warmup_frames):
                self._read_valid_frame(device)
            self._warmed_up = True
        for _ in range(self.config.flush_frames_per_capture):
            self._read_valid_frame(device)
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
        if resolution_name is not None:
            streams = getattr(device, "STREAMS", None)
            if not isinstance(streams, dict) or resolution_name not in streams:
                choices = sorted(streams) if isinstance(streams, dict) else []
                raise ValueError(
                    f"Unsupported DIGIT resolution {resolution_name!r}; "
                    f"available values: {choices}"
                )

            stream = streams[resolution_name]
            device.set_resolution(stream)

            if self.config.fps is not None:
                fps_options = (
                    stream.get("fps", {}) if isinstance(stream, dict) else {}
                )
                if self.config.fps not in fps_options:
                    raise ValueError(
                        f"Unsupported FPS {self.config.fps!r} for "
                        f"{resolution_name}; available values: "
                        f"{sorted(fps_options)}"
                    )
                device.set_fps(fps_options[self.config.fps])

        if self.config.led_intensity is not None:
            # Re-applied after any resolution/fps change rather than relying
            # on the SDK's own connect()-time default, in case a mode switch
            # disturbs it. Note: on at least this hardware/driver/kernel
            # combination this call is a confirmed no-op -- the DIGIT SDK
            # implements LED intensity as a 12-bit value packed into the UVC
            # zoom control (see Digit.set_intensity_rgb), and cv2's
            # VideoCapture.set(CAP_PROP_ZOOM, ...) silently returns False for
            # it here (verified), while even a direct V4L2 VIDIOC_S_EXT_CTRLS
            # ioctl write to the same control raises EPROTO. It's kept
            # anyway since it's harmless and may work on other setups; it is
            # NOT what fixes the green-cast startup issue (that's
            # stream_settle_delay_sec above -- a genuine sensor
            # auto-exposure warmup, unrelated to this control).
            device.set_intensity(self.config.led_intensity)

    def _require_device(self) -> DigitDeviceProtocol:
        if not self._connected or self._device is None:
            raise RuntimeError("DIGIT camera is not connected")
        return self._device

    @classmethod
    def _read_valid_frame(cls, device: DigitDeviceProtocol) -> np.ndarray:
        # The official SDK's default (transpose=False) rotates and flips the
        # sensor's native frame (e.g. 480x640x3 -> 640x480x3), which does not
        # match the orientation every tuned detector parameter and reference
        # image in this project assumes. transpose=True returns the sensor's
        # direct, un-rotated output instead -- verified against the real
        # sensor to be 480x640x3, matching the static dataset.
        frame = device.get_frame(transpose=True)
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
            from digit_interface.digit_handler import DigitHandler
        except ImportError as exc:
            raise RuntimeError(
                "The DIGIT SDK is not installed. Install it with "
                "`pip install digit-interface`."
            ) from exc

        def factory(serial: str) -> DigitDeviceProtocol:
            return DigitCamera._connect_to_streaming_node(Digit, DigitHandler, serial)

        return factory

    @staticmethod
    def _connect_to_streaming_node(
        digit_cls: Any, handler_cls: Any, serial: str
    ) -> DigitDeviceProtocol:
        """Probe every /dev/videoN node registered for this serial and
        return a fresh, not-yet-connected Digit bound to whichever one
        actually streams (the caller still calls .connect() on it -- same
        contract as returning digit_cls directly, so the normal
        connect()/_configure_stream()/_settle_stream() flow is unaffected).

        Confirmed on real hardware: this DIGIT's composite USB device
        registers two video4linux nodes under the same serial number, and
        only one of them produces frames -- the other times out on every
        single read. digit_interface.Digit's own discovery (via udev) just
        takes udev's first match, which is not stable across reconnects (it
        pointed at a different one of the two nodes across consecutive
        connections in testing, with no physical change). Digit.__init__
        only resolves dev_name via discovery and does not open the device,
        so it's safe to override dev_name before calling connect().
        """
        candidates = [
            entry["dev_name"]
            for entry in handler_cls.list_digits()
            if entry["serial"] == serial
        ]
        if not candidates:
            raise RuntimeError(f"No DIGIT sensor found with serial {serial!r}")

        last_error: Exception | None = None
        for dev_name in candidates:
            probe = digit_cls(serial)
            probe.dev_name = dev_name
            try:
                probe.connect()
                probe.get_frame(transpose=True)  # does this node stream?
            except Exception as exc:
                last_error = exc
                continue
            finally:
                try:
                    probe.disconnect()
                except Exception:
                    pass
            device = digit_cls(serial)
            device.dev_name = dev_name
            return device

        raise RuntimeError(
            f"None of the DIGIT video nodes for serial {serial!r} produced a "
            f"frame ({candidates}); last error: {last_error}"
        ) from last_error
