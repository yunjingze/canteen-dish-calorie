"""Turn the collected corrections into something actionable.

Reads data/splits/feedback/labels.csv (plus any labels_legacy_*.csv left by a
column change) and reports:

* how many corrections arrived, by mode and by whether a dish was struck out,
* the confusion pairs -- "we said pizza, it was lasagna, 4 times",
* which dishes get corrected most often, which is the shortest list for the
  next photo session.

    python scripts/feedback_report.py
    python scripts/feedback_report.py --out results/REPORT_feedback.md
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK = ROOT / "data/splits/feedback"
SPECIAL = {"not_a_dish", "not_sure"}


def rows() -> list[dict]:
    out: list[dict] = []
    for f in sorted(FEEDBACK.glob("labels*.csv")):
        with f.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                r["_file"] = f.name
                out.append(r)
    return out


def summarise(rs: list[dict]) -> dict:
    by_mode = Counter((r.get("mode") or "single") for r in rs)
    with_removal = sum(1 for r in rs if (r.get("removed") or "").strip())
    pairs: Counter = Counter()
    target: Counter = Counter()
    for r in rs:
        guessed = (r.get("predicted") or "").split(";")[0].strip()
        for c in [p.strip() for p in (r.get("corrected") or "").split(";") if p.strip()]:
            if c != guessed:
                pairs[guessed + " -> " + c] += 1
            if c not in SPECIAL:
                target[c] += 1
    struck: Counter = Counter()
    for r in rs:
        for c in [p.strip() for p in (r.get("removed") or "").split(";") if p.strip()]:
            struck[c] += 1
    return {
        "n": len(rs),
        "by_mode": dict(by_mode),
        "with_removal": with_removal,
        "confusions": pairs.most_common(),
        "corrected_targets": target.most_common(),
        "struck_out": struck.most_common(),
    }


def markdown(s: dict) -> str:
    lines = [
        "# 反馈汇总",
        "",
        "共 **%d** 条人工纠正；按模式：%s；其中 %d 条包含“划掉某样菜”。"
        % (s["n"], s["by_mode"], s["with_removal"]),
        "",
    ]
    if not s["confusions"]:
        lines += ["还没有“说错→纠正”的配对。", ""]
    else:
        lines += ["## 最常被认错的", "", "| 我们说 | 实际是 | 次数 |", "|---|---|---|"]
        for pair, n in s["confusions"][:15]:
            said, real = pair.split(" -> ")
            lines.append("| %s | %s | %d |" % (said, real, n))
        lines.append("")
    if s["struck_out"]:
        lines += ["## 被划掉（其实不在盘里）", "", "| 菜 | 次数 |", "|---|---|"]
        for c, n in s["struck_out"]:
            lines.append("| %s | %d |" % (c, n))
        lines.append("")
    if s["corrected_targets"]:
        lines += [
            "## 需要补拍的（按被纠正次数）",
            "",
            "这些是真值里反复出现的菜，说明模型在这几类上最容易错：",
            "",
        ]
        for c, n in s["corrected_targets"][:10]:
            lines.append("- %s：%d 次" % (c, n))
        lines.append("")
    lines += [
        "## 怎么用",
        "",
        "1. 同一个“说错→纠正”出现 3 次以上：先看那几张图，确认是标注错、类内差异还是域偏移；",
        "2. 纠正目标集中在食堂常见的几类：优先补那几类的实拍；",
        "3. 划掉次数多的菜：检查多标签阈值，或该类在托盘上被过度触发。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/REPORT_feedback.md")
    ap.add_argument("--json", default="results/tables/feedback_report.json")
    args = ap.parse_args()

    s = summarise(rows())
    md = markdown(s)
    (ROOT / args.out).write_text(md, encoding="utf-8")
    (ROOT / args.json).write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)
    print("wrote %s and %s" % (args.out, args.json))


if __name__ == "__main__":
    main()
