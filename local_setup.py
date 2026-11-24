from __future__ import annotations
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent


def enable_modules():
    config = Path(os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "ultralytics_config")))
    config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    import ultralytics.nn as nn

    folder = ROOT / "custom" / "nn"
    if str(folder) not in nn.__path__:
        nn.__path__.insert(0, str(folder))
