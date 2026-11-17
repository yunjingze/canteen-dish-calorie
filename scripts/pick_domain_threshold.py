# -*- coding: utf-8 -*-
"""Re-sweep the domain-probe threshold from the rows eval_domain_gate.py saved.

The probe is a pure veto (accept_on = accept_off & p_domain >= t), so every
threshold can be scored without another forward pass.  Prints the trade-off the
user actually sees; the deployed threshold stays the one fitted from the 5th
percentile of the Chinese validation split (same false-alarm budget as the
other gates), and this table only says what a stricter or looser one would buy.

Run: python scripts/pick_domain_threshold.py [--floor 0.90]
Writes results/tables/domain_head_choice.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAGS = ["in", "out_cn", "out_ood", "out_other", "real"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="results/tables/domain_head.json")
    args = ap.parse_args()
    head_t = float(json.loads((ROOT / args.head).read_text("utf-8"))["threshold"])

    rows = {}
    for tag in TAGS:
        p = ROOT / f"results/tables/_domain_gate_{tag}.npz"
        if not p.is_file():
            raise SystemExit(f"missing {p}; run scripts/eval_domain_gate.py first")
        z = np.load(p, allow_pickle=True)
        rows[tag] = {"accept_off": z["accept_off"].astype(bool),
                     "p": z["p_domain"].astype(np.float64)}

    grid = sorted(set(np.round(list(np.arange(0.05, 0.96, 0.05)) + [head_t], 2).tolist()))
    table = []
    for t in grid:
        acc = {tag: float(np.mean(r["accept_off"] & (r["p"] >= t))) for tag, r in rows.items()}
        table.append({"t": float(t), "accept": {k: round(v, 4) for k, v in acc.items()},
                      "vetoed_by_probe": {
                          tag: round(float(np.mean(r["accept_off"] & (r["p"] < t))), 4)
                          for tag, r in rows.items()}})
        a = table[-1]["accept"]
        mark = "  <- deployed" if abs(float(t) - head_t) <= 0.0051 else ""
        print(f"t={t:.2f}  in {a['in']*100:5.1f}  out_cn {a['out_cn']*100:5.1f}  "
              f"out_ood {a['out_ood']*100:5.1f}  out_other {a['out_other']*100:5.1f}  "
              f"real {a['real']*100:5.1f}{mark}", flush=True)

    out = {"head_threshold": head_t, "sets": TAGS, "sweep": table}
    (ROOT / "results/tables/domain_head_choice.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), "utf-8")
    print("wrote results/tables/domain_head_choice.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
