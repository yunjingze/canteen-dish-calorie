"""Collect the three 116-class runs scored by eval_wide_candidate.py into one table."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "deployed_f101_head": "results/tables/wide_candidate_deployed.json",
    "f101n_only": "results/tables/wide_candidate_f101n.json",
    "f101n_backbone_f101_head": "results/tables/wide_candidate_hybrid.json",
    "f101n_hybrid_e10": "results/tables/wide_candidate_hybrid_e10.json",
}
ROWS = [
    ("wide test top-1", lambda d: d["test"]["top1"]),
    ("wide test top-5", lambda d: d["test"]["top5"]),
    ("wide test macro F1", lambda d: d["test_metrics"]["macro_f1"]),
    ("wide test ECE", lambda d: d["test_metrics"]["ece"]),
    ("food101 group top-1", lambda d: d["test_groups"]["food101"]["top1"]),
    ("chinese group top-1", lambda d: d["test_groups"]["chinese"]["top1"]),
    ("course test r20", lambda d: d["course_test"]["top1_restricted"]),
    ("course test full116", lambda d: d["course_test"]["top1_full116"]),
    ("canteen clean r20", lambda d: d["canteen_clean"]["top1_restricted"]),
    ("canteen clean full116", lambda d: d["canteen_clean"]["top1_full116"]),
    ("canteen proxy r20", lambda d: d["canteen_proxy"]["top1_restricted"]),
    ("canteen proxy full116", lambda d: d["canteen_proxy"]["top1_full116"]),
    ("canteen real r20", lambda d: d["canteen_real"]["top1_restricted"]),
    ("canteen real full116", lambda d: d["canteen_real"]["top1_full116"]),
]

data = {}
for key, rel in FILES.items():
    p = ROOT / rel
    if p.is_file():
        data[key] = json.loads(p.read_text("utf-8"))

out = {"runs": {}, "table": {}}
for key, d in data.items():
    out["runs"][key] = {k: d.get(k) for k in ("ckpt", "epoch", "temperature", "val_acc", "val_macro_f1")}
    out["table"][key] = {name: round(float(fn(d)) * 100, 2) for name, fn in ROWS}

out_path = ROOT / "results/tables/wide_backbone_compare.json"
out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

hdr = "| metric | " + " | ".join(data) + " |"
sep = "| --- | " + " | ".join("---" for _ in data) + " |"
lines = [hdr, sep]
for name, _ in ROWS:
    cells = [f"{out['table'][k][name]:.2f}" for k in data]
    lines.append(f"| {name} | " + " | ".join(cells) + " |")
print("\n".join(lines))
print()
print("runs:", json.dumps(out["runs"], ensure_ascii=False))
