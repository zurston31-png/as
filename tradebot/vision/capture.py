"""Screen capture of the chart window.

This is the backup path. Structured data is always preferred - reading candle
positions out of a bitmap throws away precision you already had. Capture earns
its place in two situations: the data feed has gone quiet and you want to know
whether the chart is still moving, and cross-checking that the symbol on screen
is actually the symbol the bot thinks it is trading.

`mss` and `pillow` are optional extras; everything degrades to "vision
unavailable" if they aren't installed.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import VisionConfig
from ..models import utcnow

log = logging.getLogger(__name__)

try:
    import mss  # type: ignore
except ImportError:  # pragma: no cover
    mss = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


@dataclass
class Capture:
    png: bytes
    width: int
    height: int
    ts: datetime
    path: Optional[str] = None

    @property
    def base64(self) -> str:
        return base64.standard_b64encode(self.png).decode("ascii")


class ScreenCapture:
    def __init__(self, config: VisionConfig) -> None:
        self.cfg = config
        self._last_at: float = 0.0

    @staticmethod
    def availability() -> dict[str, object]:
        return {
            "mss_installed": mss is not None,
            "pillow_installed": Image is not None,
            "ready": mss is not None and Image is not None,
        }

    def available(self) -> bool:
        return mss is not None and Image is not None

    def grab(self) -> Optional[Capture]:
        """Capture the configured region. Returns None if capture isn't possible."""
        if not self.available():
            log.debug("screen capture unavailable: install `mss` and `pillow`")
            return None
        try:
            with mss.mss() as sct:
                region = self.cfg.region
                if region:
                    monitor = {
                        "left": int(region["left"]),
                        "top": int(region["top"]),
                        "width": int(region["width"]),
                        "height": int(region["height"]),
                    }
                else:
                    index = min(self.cfg.monitor, len(sct.monitors) - 1)
                    monitor = sct.monitors[index]
                shot = sct.grab(monitor)
                img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        except Exception as exc:  # noqa: BLE001 - headless box, denied permission, etc.
            log.warning("screen capture failed: %s", exc)
            return None

        if img.width > self.cfg.max_width:
            ratio = self.cfg.max_width / img.width
            img = img.resize((self.cfg.max_width, int(img.height * ratio)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        cap = Capture(buf.getvalue(), img.width, img.height, utcnow())

        if self.cfg.save_captures:
            cap.path = self._save(cap)
        return cap

    def _save(self, cap: Capture) -> Optional[str]:
        try:
            directory = Path(self.cfg.capture_dir)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"chart_{cap.ts.strftime('%Y%m%d_%H%M%S')}.png"
            path.write_bytes(cap.png)
            return str(path)
        except OSError as exc:
            log.warning("could not save capture: %s", exc)
            return None
