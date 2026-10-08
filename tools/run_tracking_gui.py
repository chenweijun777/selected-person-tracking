"""Launch the video picker without requiring command-line arguments."""

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs/.ultralytics"))
(ROOT / "outputs").mkdir(exist_ok=True)
log = ROOT / "outputs/tracking_gui.log"
logging.basicConfig(
    filename=log,
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s %(levelname)s %(message)s",
)
if sys.stdout is None:
    sys.stdout = log.open("a", encoding="utf-8", buffering=1)
if sys.stderr is None:
    sys.stderr = sys.stdout

from a1_tracker.tracking_gui import main

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger(__name__).exception("GUI startup failed")
        raise
