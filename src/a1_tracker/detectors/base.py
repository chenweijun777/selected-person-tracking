from __future__ import annotations

from typing import Protocol

from a1_tracker.types import Detection, FramePacket


class Detector(Protocol):
    """检测后端统一接口；输出框必须还原到原图 xyxy 坐标。"""

    def detect(self, frame: FramePacket) -> list[Detection]: ...
