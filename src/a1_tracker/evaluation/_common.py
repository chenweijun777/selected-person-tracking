"""Shared I/O helpers. Never write to input paths."""

import csv
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[3]


def canonical_split(value):
    """Normalize the frozen protocol's legacy alias to the current split name."""
    return "validation" if value == "historical" else value


GLOBAL_FIELDS = ("HOTA", "IDF1", "IDSW", "Frag", "AssA", "AssRe", "AssPr", "DetA", "DetRe", "MOTA")
EVENT_FIELDS = (
    "event_id",
    "sequence",
    "gt_id",
    "event_type",
    "last_visible_frame",
    "reappearance_frame",
    "main_visible_frame",
    "pre_event_pred_id",
    "post_event_pred_id",
    "same_id_recovered",
    "new_id",
    "wrong_id",
    "no_output",
    "recovery_success",
    "recovery_delay_frames",
    "recovery_delay_seconds",
    "eligible",
    "exclusion_reason",
    "provisional",
)


def resolve_asset(value, *, required=True):
    """Read an asset relative to the project root."""
    key = str(value).replace("\\", "/")
    original = Path(key)
    if original.is_absolute() or PureWindowsPath(key).is_absolute():
        if required:
            raise FileNotFoundError(f"Asset path must be project-relative: {value}")
        return None
    path = (ROOT / original).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        if required:
            raise FileNotFoundError(f"Asset is outside the project: {value}")
        return None
    if path.is_file():
        return path
    if required:
        raise FileNotFoundError(f"Missing frozen asset: {value}; prepare local reproduction inputs")
    return None


def project_path(value):
    """Resolve project paths independently of the caller's working directory."""
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def portable_record(value):
    """Serialize in-project report paths relative to the repository."""
    if isinstance(value, dict):
        return {key: portable_record(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_record(item) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute() and path.is_relative_to(ROOT):
            return path.relative_to(ROOT).as_posix()
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=()):
    names = list(dict.fromkeys([*fields, *(k for row in rows for k in row)]))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def boolean(value):
    if str(value).lower() not in ("true", "false"):
        raise ValueError(f"Expected true/false, got {value!r}")
    return str(value).lower() == "true"


def box(value):
    if isinstance(value, str):
        value = json.loads(value)
    if value is None or len(value) != 4:
        raise ValueError("valid prediction requires bbox_xyxy with four coordinates")
    result = tuple(float(x) for x in value)
    if (
        not all(math.isfinite(x) for x in result)
        or result[2] <= result[0]
        or result[3] <= result[1]
    ):
        raise ValueError(f"Invalid bbox: {value}")
    return result
