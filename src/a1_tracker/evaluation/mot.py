"""Official TrackEval adapter."""

from __future__ import annotations

import configparser
import contextlib
import json
from pathlib import Path

import trackeval

from ._common import ROOT

GT = ROOT / "data/MOT17/MOT17/train"
OUT = ROOT / "outputs/tracker_evaluation"


def run(
    short_sequences: list[str],
    groups: list[str],
    label: str,
    *,
    tracks_root: Path | None = None,
    output_root: Path | None = None,
    gt_root: Path | None = None,
) -> dict:
    gt_root = gt_root or GT
    names = [f"MOT17-{short}-SDP" for short in short_sequences]
    lengths = {}
    for name in names:
        ini = configparser.ConfigParser()
        ini.read(gt_root / name / "seqinfo.ini")
        lengths[name] = ini.getint("Sequence", "seqLength")
    directory = output_root or OUT / "trackeval" / label
    directory.mkdir(parents=True, exist_ok=True)
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "PRINT_CONFIG": False,
            "PLOT_CURVES": False,
            "TIME_PROGRESS": False,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
        }
    )
    data_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    data_config.update(
        {
            "GT_FOLDER": str(gt_root),
            "TRACKERS_FOLDER": str(tracks_root or OUT / "tracks"),
            "OUTPUT_FOLDER": str(directory),
            "TRACKERS_TO_EVAL": groups,
            "TRACKER_SUB_FOLDER": "",
            "OUTPUT_SUB_FOLDER": "",
            "SEQ_INFO": lengths,
            "SKIP_SPLIT_FOL": True,
            "PRINT_CONFIG": False,
            "CLASSES_TO_EVAL": ["pedestrian"],
            "DO_PREPROC": True,
        }
    )
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    with (
        (directory / "raw_stdout.txt").open("w", encoding="utf-8") as file,
        contextlib.redirect_stdout(file),
    ):
        result, messages = trackeval.Evaluator(eval_config).evaluate(
            [trackeval.datasets.MotChallenge2DBox(data_config)], metrics
        )
    import numpy as np

    def serializable(value):
        if isinstance(value, dict):
            return {str(key): serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [serializable(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    (directory / "raw_results.json").write_text(
        json.dumps({"results": serializable(result), "messages": serializable(messages)}, indent=2),
        encoding="utf-8",
    )
    for group in groups:
        summary = directory / group / "pedestrian_summary.txt"
        print(group, summary.read_text(encoding="utf-8") if summary.exists() else "no summary")
    return serializable({"results": result, "messages": messages})
