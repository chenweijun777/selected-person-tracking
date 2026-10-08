from __future__ import annotations

import argparse
import json

from a1_tracker.target import TargetManager
from a1_tracker.trackers import GreedyIoUTracker
from a1_tracker.types import Detection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="指定行人跟踪项目的基础组件安装检查")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    return parser


def run_smoke() -> dict[str, object]:
    tracker = GreedyIoUTracker(match_iou=0.2, max_missed_frames=3)
    manager = TargetManager(lost_timeout_frames=5, verify_observations=2, candidate_iou=0.1)
    tracks = tracker.update([Detection((10, 10, 50, 100), 0.9)])
    manager.select_track(tracks[0])
    result = manager.update(tracks, timestamp=0.0)
    return {
        "ok": result.valid,
        "state": result.state.value,
        "tracker_id": result.tracker_id,
        "session_target_id_created": result.session_target_id is not None,
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_smoke()
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print("Tracking component smoke check:", result)


if __name__ == "__main__":
    main()
