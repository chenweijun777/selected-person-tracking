"""Run the formal fine-tuned 26s + ShortLink + standalone A2 application."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs/.ultralytics"))

from a1_tracker.video_app import main

if __name__ == "__main__":
    main(joint=True)
