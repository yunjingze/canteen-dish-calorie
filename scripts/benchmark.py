"""Params / MACs / latency for every trained checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from foodcls.utils import count_macs, count_params, measure_latency, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", default=str(ROOT / "results/tables/complexity.json"))
    args = ap.parse_args()

    runs = args.runs or [str(p) for p in sorted((ROOT / "results/runs").iterdir()) if (p / "best.pt").exists()]
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for run in runs:
        path = Path(run)
        ckpt = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
        model = build_model(ckpt["arch"], len(ckpt.get("class_names") or CLASS_NAMES), pretrained=False)
        model.load_state_dict(ckpt["model"])
        model_cpu = model.eval()
        model_gpu = build_model(ckpt["arch"], len(CLASS_NAMES), pretrained=False)
        model_gpu.load_state_dict(ckpt["model"])
        model_gpu = model_gpu.to(gpu).eval()

        row = {
            "run": path.name,
            "arch": ckpt["arch"],
            "mode": ckpt["mode"],
            "color_aug": ckpt.get("color_aug", False),
            "params_m": count_params(model_cpu) / 1e6,
            "macs_g": count_macs(model_gpu, gpu),
            **{f"gpu_{k}": v for k, v in measure_latency(model_gpu, gpu).items()},
            **{f"cpu_{k}": v for k, v in measure_latency(model_cpu, torch.device("cpu"), runs=20).items()},
        }
        rows.append(row)
        print(json.dumps(row, indent=2))
        del model_gpu
        torch.cuda.empty_cache()

    save_json(args.out, rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
