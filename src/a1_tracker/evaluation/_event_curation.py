"""Apply the user's event exclusions without changing frozen experiment inputs."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXCLUSIONS = ROOT / "manifests/recovery_event_exclusions_20261005.json"


def excluded_event_ids(path=EXCLUSIONS):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported event exclusion schema")
    ids = [row["event_id"] for row in payload["decisions"]]
    if len(ids) != len(set(ids)) or any(not eid for eid in ids):
        raise ValueError("Duplicate or empty excluded event ID")
    if any(row["verdict"] != "exclude_no_reference_value" for row in payload["decisions"]):
        raise ValueError("Unexpected event exclusion verdict")
    return set(ids)


def active_events(rows, excluded=None):
    excluded = excluded_event_ids() if excluded is None else excluded
    return [row for row in rows if row["event_id"] not in excluded]
