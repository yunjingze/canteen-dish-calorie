"""Run the full experiment matrix (6 configurations) then benchmark them."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

EXPERIMENTS = [
    ("resnet18_frozen_noaug", ["--arch", "resnet18", "--mode", "frozen", "--epochs", "12"]),
    ("resnet18_frozen_aug", ["--arch", "resnet18", "--mode", "frozen", "--epochs", "12", "--color-aug"]),
    ("resnet18_last_stage_noaug", ["--arch", "resnet18", "--mode", "last_stage", "--epochs", "12"]),
    ("resnet18_last_stage_aug", ["--arch", "resnet18", "--mode", "last_stage", "--epochs", "12", "--color-aug"]),
    ("mobilenet_v3_small_last_stage_aug", ["--arch", "mobilenet_v3_small", "--mode", "last_stage", "--epochs", "12", "--color-aug"]),
    ("resnet18_scratch_noaug", ["--arch", "resnet18", "--mode", "full", "--epochs", "12", "--no-pretrained"]),
]


def main():
    only = sys.argv[1:] or None
    for name, extra in EXPERIMENTS:
        if only and name not in only:
            continue
        out = ROOT / "results/runs" / name
        print(f"=== {name} ===", flush=True)
        cmd = [PY, "-m", "foodcls.train", "--data-root", "../data", "--out", str(out), "--tag", name, *extra]
        subprocess.run(cmd, cwd=ROOT / "src", check=True)
    subprocess.run([PY, str(ROOT / "scripts/benchmark.py")], cwd=ROOT, check=True)
    subprocess.run([PY, str(ROOT / "scripts/make_figures.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
