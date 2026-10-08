"""Frame-local identity handoffs from successful ShortLink stitches."""

from dataclasses import dataclass

from a1_tracker.types import BBox, validate_bbox


@dataclass(frozen=True)
class IdentityRemap:
    frame_id: int
    young_id: int
    old_id: int
    bbox_xyxy: BBox

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.frame_id, self.young_id, self.old_id)):
            raise ValueError("Identity remap requires positive integer frame and IDs")
        if self.young_id == self.old_id:
            raise ValueError("Identity remap requires distinct IDs")
        validate_bbox(self.bbox_xyxy)


@dataclass(frozen=True)
class IdentityFrame:
    generation: int
    frame_id: int
    remaps: tuple[IdentityRemap, ...] = ()
