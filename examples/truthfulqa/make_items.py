"""Build the sample: 200 TruthfulQA questions, each with one candidate answer to judge.

TruthfulQA (Lin, Hilton and Evans, 2021, arXiv:2109.07958) is Apache-2.0:
https://github.com/sylinrl/TruthfulQA

For each sampled question the candidate answer is, with equal chance, one of
the dataset's correct answers or one of its incorrect answers, picked at
random. The reference is the dataset's "Best Answer". The extra fields
(category, label) are ignored by jj.

    python examples/truthfulqa/make_items.py            # downloads the CSV
    python examples/truthfulqa/make_items.py local.csv  # or uses a local copy
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
SEED = 2026
N = 200
OUT = Path(__file__).with_name("items.jsonl")


def split_answers(cell: str) -> list[str]:
    return [a.strip() for a in (cell or "").split(";") if a.strip()]


def main() -> None:
    if len(sys.argv) > 1:
        raw = Path(sys.argv[1]).read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=60) as r:
            raw = r.read()
    print(f"TruthfulQA.csv sha256 {hashlib.sha256(raw).hexdigest()}")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    rng = random.Random(SEED)
    picked = sorted(rng.sample(range(len(rows)), N))
    with OUT.open("w", encoding="utf-8") as f:
        for idx in picked:
            row = rows[idx]
            correct = split_answers(row["Correct Answers"])
            incorrect = split_answers(row["Incorrect Answers"])
            coin = rng.random() < 0.5
            use_correct = coin and bool(correct)
            pool = correct if use_correct else incorrect
            item = {
                "id": f"tqa-{idx:03d}",
                "input": row["Question"].strip(),
                "output": rng.choice(pool),
                "reference": row["Best Answer"].strip(),
                "category": row.get("Category", "").strip(),
                "label": "correct" if use_correct else "incorrect",
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {N} items from {len(rows)} questions to {OUT}")


if __name__ == "__main__":
    main()
