"""Shared helpers: seeding, metrics, complexity and timing."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (labels, probs, preds) over a loader."""
    model.eval()
    labels, probs = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        labels.append(y.numpy())
    labels = np.concatenate(labels)
    probs = np.concatenate(probs)
    return labels, probs, probs.argmax(1)


def classification_metrics(labels: np.ndarray, probs: np.ndarray, class_names: Sequence[str]) -> Dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    preds = probs.argmax(1)
    p, r, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(class_names))), zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
        "n": int(len(labels)),
    }


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(labels)) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def count_macs(model: torch.nn.Module, device, size: int = 224) -> float:
    """MACs in Giga (1 MAC = 2 FLOPs)."""
    from thop import profile

    dummy = torch.randn(1, 3, size, size, device=device)
    macs, _ = profile(model, inputs=(dummy,), verbose=False)
    return float(macs) / 1e9


@torch.no_grad()
def measure_latency(model, device, size: int = 224, warmup: int = 10, runs: int = 50) -> Dict[str, float]:
    """Single-image latency (ms) with warmup, plus throughput."""
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts: List[float] = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts_arr = np.asarray(ts)
    return {
        "latency_ms_mean": float(ts_arr.mean()),
        "latency_ms_p50": float(np.percentile(ts_arr, 50)),
        "latency_ms_p95": float(np.percentile(ts_arr, 95)),
        "fps": float(1000.0 / ts_arr.mean()),
    }


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
