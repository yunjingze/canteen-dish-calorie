"""Confidence intervals and paired tests for the numbers we quote.

Every headline accuracy so far is a point estimate on a fixed split, and the
splits are small (1,000 proxy images, 100 real photos).  This script recomputes
both the intervals and the paired tests from the cached per-view logits
(results/_ens_<run>__<split>.npz), so it needs no GPU and no data pass.

Two fusions are reported:

* "single" -- view 0 only, the centre crop at short side 256.  This is the
  protocol behind Table II of the paper and the numbers in the README.
* "six"    -- mean of the six view logits, which is what the web demo scores.

Primary claims use "single" (it is the protocol of the table); "six" is quoted
only where the deployed behaviour matters.

Outputs results/tables/stats.json and prints a markdown block.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"

RUNS = {
    "E4": "resnet18_last_stage_aug",
    "E7": "resnet18_last_stage_aug_f101n",
    "E8": "resnet18_last_stage_aug_f101n_nodedup",
    "E12": "resnet18_last_stage_aug_f101n_e24",
    "seed7": "resnet18_last_stage_aug_seed7",
}
SPLITS = ("test", "canteen", "canteen_s7")
B = 4000


def load_split(run: str, split: str):
    """Correctness under both fusions for every image of one split."""
    data = np.load(ROOT / "results" / f"_ens_{run}__{split}.npz", allow_pickle=True)
    logits = data["logits"]                     # (N, 6, 20)
    labels = data["labels"]
    single = logits[:, 0].argmax(axis=1) == labels
    six = logits.mean(axis=1).argmax(axis=1) == labels
    return single, six


def bootstrap_ci(correct: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(correct)
    draws = correct[rng.integers(0, n, size=(B, n))].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(correct.mean()), float(low), float(high)


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    if only_a + only_b == 0:
        return {"only_a": 0, "only_b": 0, "p": 1.0}
    p = float(binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue)
    return {"only_a": only_a, "only_b": only_b, "p": p}


def ensemble_correct(split: str) -> np.ndarray | None:
    """Mean softmax over every cached run of this split (the "five models" row)."""
    probs, labels = [], None
    for run in RUNS.values():
        path = ROOT / "results" / f"_ens_{run}__{split}.npz"
        if not path.is_file():
            continue
        data = np.load(path, allow_pickle=True)
        logits = data["logits"][:, 0]           # keep the single-view protocol
        logits = logits - logits.max(axis=1, keepdims=True)
        probs.append(np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True))
        labels = data["labels"]
    if not probs:
        return None
    return np.mean(probs, axis=0).argmax(axis=1) == labels


def real_photos(fusion: str) -> dict:
    raw = json.loads((TABLES / "real_photos_views.json").read_text(encoding="utf-8"))
    rows = raw["single_dish"]["rows"]
    key = {"single": "single", "six": "fused"}[fusion]   # the json calls it "fused"
    out = {}
    for tag in ("E4", "E7", "E8", "E12", "E13"):
        ok = np.array([r[tag][key] == r["label"] for r in rows if tag in r])
        if ok.size:
            out[tag] = ok
    return out


def gate() -> dict:
    """Rejection on the negatives that are still negatives.

    api_sweep.json was written before the seven mislabelled "negatives" (bread
    rolls, a tray of salads, a cone of food) were moved to food_nonvocab, so the
    sweep still lists 138 rows; drop those seven.
    """
    rows = json.loads((TABLES / "api_sweep.json").read_text(encoding="utf-8"))
    relabelled = json.loads((ROOT / "data/splits/negatives/v1_categories.json").read_text(encoding="utf-8"))
    food = {name for name, cat in relabelled.items() if cat == "food_nonvocab"}
    keep = lambda r: Path(r["file"]).name not in food
    neg = np.array([not r["in_scope"] for r in rows
                    if r["kind"] == "neg" and r.get("food_prob") is not None and keep(r)])
    real = np.array([r["in_scope"] for r in rows if r["kind"] == "real"])
    return {"negatives_rejected": neg, "real_kept": real}


def main() -> None:
    out: dict = {"single": {}, "six": {}, "mcnemar_single": {}, "mcnemar_six": {}, "gate": {}}
    correct = {"single": {}, "six": {}}

    for split in SPLITS:
        for tag, run in RUNS.items():
            if not (ROOT / "results" / f"_ens_{run}__{split}.npz").is_file():
                continue
            for fusion, ok in zip(("single", "six"), load_split(run, split)):
                correct[fusion][(tag, split)] = ok
                acc, low, high = bootstrap_ci(ok)
                out[fusion][f"{tag}/{split}"] = {"acc": acc, "lo": low, "hi": high, "n": int(ok.size)}
        ok = ensemble_correct(split)
        if ok is not None:
            correct["single"][("all5", split)] = ok
            acc, low, high = bootstrap_ci(ok)
            out["single"][f"all5/{split}"] = {"acc": acc, "lo": low, "hi": high, "n": int(ok.size)}

    pairs = [("E4", "E7"), ("E7", "E8"), ("E7", "all5"), ("E8", "all5")]
    for fusion, bucket in (("single", "mcnemar_single"), ("six", "mcnemar_six")):
        for a, b in pairs:
            for split in SPLITS:
                if (a, split) in correct[fusion] and (b, split) in correct[fusion]:
                    out[bucket][f"{a}-vs-{b}/{split}"] = mcnemar(correct[fusion][(a, split)],
                                                                 correct[fusion][(b, split)])

    for fusion in ("single", "six"):
        real = real_photos(fusion)
        out.setdefault("real_" + fusion, {})
        for tag, ok in real.items():
            acc, low, high = bootstrap_ci(ok)
            out["real_" + fusion][tag] = {"acc": acc, "lo": low, "hi": high, "n": int(ok.size)}
        if "E4" in real and "E7" in real:
            out["mcnemar_" + fusion]["E4-vs-E7/real"] = mcnemar(real["E4"], real["E7"])
        if "E7" in real and "E8" in real:
            out["mcnemar_" + fusion]["E7-vs-E8/real"] = mcnemar(real["E7"], real["E8"])

    g = gate()
    for key, arr in g.items():
        acc, low, high = bootstrap_ci(arr)
        out["gate"][key] = {"acc": acc, "lo": low, "hi": high, "n": int(arr.size)}

    (TABLES / "stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    def table(bucket: str, title: str) -> None:
        print(f"## {title}\n")
        print("| run/split | acc | 95% CI | n |")
        print("| --- | ---: | :---: | ---: |")
        for key, v in out[bucket].items():
            print(f"| {key} | {100*v['acc']:.2f} | [{100*v['lo']:.2f}, {100*v['hi']:.2f}] | {v['n']} |")
        print()

    table("single", "single view (Table II protocol), accuracy [95% CI]")
    table("six", "six-view fusion (what the demo scores)")
    print("## paired tests, single view (exact McNemar)\n")
    for key, v in out["mcnemar_single"].items():
        print(f"- {key}: only A {v['only_a']}, only B {v['only_b']}, p = {v['p']:.4f}")
    print("\n## paired tests, six views\n")
    for key, v in out["mcnemar_six"].items():
        print(f"- {key}: only A {v['only_a']}, only B {v['only_b']}, p = {v['p']:.4f}")
    print("\n## real photos, single / six\n")
    for fusion in ("single", "six"):
        for tag, v in out["real_" + fusion].items():
            print(f"- {tag}/{fusion}: {100*v['acc']:.1f}% [{100*v['lo']:.1f}, {100*v['hi']:.1f}], n = {v['n']}")
    print("\n## gate\n")
    for key, v in out["gate"].items():
        print(f"- {key}: {100*v['acc']:.1f}% [{100*v['lo']:.1f}, {100*v['hi']:.1f}], n = {v['n']}")
    print(f"\nwrote {TABLES / 'stats.json'}")


if __name__ == "__main__":
    main()
