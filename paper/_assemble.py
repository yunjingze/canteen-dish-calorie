"""Assemble paper/main.tex from paper/sections/*.tex.

The paper is edited as one file per section; this script glues them in reading
order so that the LaTeX that gets compiled is always the sum of the parts and
never a hand-merged copy. Run it before pdflatex.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SECTIONS = HERE / "sections"

# Reading order. The canteen experiments come first because the canteen branch
# is the paper's main line; the Western baseline follows as the yardstick.
ORDER = [
    "00_preamble.tex",
    "01_abstract.tex",
    "02_intro.tex",
    "03_related.tex",
    "04_dataset.tex",
    "05_method.tex",
    "06_experiments.tex",
    "06_baseline.tex",
    "07_discussion.tex",
    "08_conclusion.tex",
]


def main() -> None:
    missing = [name for name in ORDER if not (SECTIONS / name).exists()]
    if missing:
        raise SystemExit(f"missing sections: {missing}")
    chunks = []
    for name in ORDER:
        text = (SECTIONS / name).read_text(encoding="utf-8").strip("\n")
        chunks.append(text)
    out = HERE / "main.tex"
    body = "\n\n".join(chunks) + "\n"
    out.write_text(body, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(body)} chars, {body.count(chr(10)) + 1} lines)")


if __name__ == "__main__":
    main()
