"""Campus canteen dish recognition with calorie-range estimation."""
from __future__ import annotations

import os
from pathlib import Path

# Keep torchvision's pretrained weights inside the project so that the run is
# reproducible even when the user profile cache is not writable.
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[2] / ".torch"))

__version__ = "1.0.0"
