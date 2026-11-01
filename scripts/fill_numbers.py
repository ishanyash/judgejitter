"""Write the measured sample numbers into README.md and LAUNCH.md.

Every number comes straight from the stats JSON that `jj report --json`
wrote, so the docs match the run exactly.

    python scripts/fill_numbers.py sample/stats.json README.md LAUNCH.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

START = "<!-- sample-numbers:start -->"
END = "<!-- sample-numbers:end -->"


def pct(x: float | None, nd: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{nd}f}%"


def block(d: dict) -> str:
    run, tot, ag, pr = d["run"], d["totals"], d["agreement"], d["pass_rate"]
    n, k = tot["items"], run["k"]
    s, m = pr["single_run"], pr["majority_of_k"]
    full = pr["majority_of_k_with_item_sampling"]
    settings = run.get("settings") or {}
    sampling = (
        "no sampling parameters (model default)"
        if settings.get("temperature") is None
        else f"temperature {settings['temperature']}"
    )
    rec = pr["recommended_k"]
    rows = [
        ("Calls", f"{tot['calls']:,} ({tot['failed_calls']} failed or unreadable)"),
        ("Fleiss' kappa (self-agreement)", "n/a" if ag["fleiss_kappa"] is None else f"{ag['fleiss_kappa']:.3f}"),
        (f"Items with all {k} verdicts the same", f"{n - ag['items_flipped']} of {n} ({pct(ag['items_all_agree'])})"),
        ("Items that flipped at least once", f"{ag['items_flipped']} of {n} ({pct(ag['items_flipped'] / n)})"),
        ("Mean flip rate", pct(ag["mean_flip_rate"])),
        (f"The {len(pr['per_repeat'])} single runs scored", ", ".join(pct(x) for x in pr["per_repeat"])),
        ("Single run, 95% of reruns", f"{pct(s['low'])} to {pct(s['high'])} ({s['width_points']:.1f} points)"),
        (
            f"Majority of {m['k']}, 95% of reruns",
            f"{pct(m['low'])} to {pct(m['high'])} ({m['width_points']:.1f} points)",
        ),
        (
            f"Calls per item for a {pr['target_width_points']:g}-point interval",
            str(rec) if rec else "more than 51",
        ),
        (
            f"With item sampling ({n} items), majority of {m['k']}",
            f"{pct(full['low'])} to {pct(full['high'])} ({full['width_points']:.1f} points)",
        ),
        ("Tokens", f"{tot['input_tokens']:,} in, {tot['output_tokens']:,} out"),
        ("Cost", "unknown" if tot["cost_usd"] is None else f"${tot['cost_usd']:.4f}"),
    ]
    lines = [
        START,
        f"Measured {(run.get('finished_at') or run.get('created_at') or '')[:10]}, run `{run['run_id']}`: "
        f"`{run['model']}` as judge, API version {run['api_version']}, {sampling}, "
        f"{'structured JSON output' if run.get('structured_output') else 'free-text output'}, "
        f"{n} TruthfulQA items x {k} calls.",
        "",
        "| Measure | Result |",
        "| --- | --- |",
        *[f"| {a} | {b} |" for a, b in rows],
        "",
        "Cost is the token counts the API returned, priced at list price. "
        "The raw numbers are in [sample/stats.json](sample/stats.json).",
        END,
    ]
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: fill_numbers.py STATS.json FILE.md [FILE.md ...]")
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    text = block(data)
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    for name in sys.argv[2:]:
        path = Path(name)
        doc = path.read_text(encoding="utf-8")
        if not pattern.search(doc):
            sys.exit(f"{name}: no {START} ... {END} block")
        path.write_text(pattern.sub(lambda _: text, doc), encoding="utf-8")
        print(f"updated {name}")


if __name__ == "__main__":
    main()
