"""Turn a stored run into numbers (JSON) and one self-contained HTML page."""

from __future__ import annotations

import html
import json
import math
from collections import Counter
from importlib import resources

from jinja2 import Environment
from markupsafe import Markup

from . import __version__
from .stats import Interval, RunStats, summarise
from .store import RunData

REPO_URL = "https://github.com/ishanyash/judgejitter"


def analyse(run: RunData, *, target_points: float = 2.0, n_boot: int = 4000, seed: int = 0) -> RunStats:
    return summarise(
        run.verdict_lists(),
        scale=run.meta["scale"],
        pass_threshold=run.meta.get("pass_threshold") or 4,
        target_points=target_points,
        n_boot=n_boot,
        seed=seed,
    )


def _r(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), nd)


def _iv(iv: Interval) -> dict:
    return {
        "k": iv.k,
        "pass_rate": _r(iv.estimate),
        "low": _r(iv.low),
        "high": _r(iv.high),
        "width_points": _r(iv.width_points, 2),
    }


def unstable_items(run: RunData, st: RunStats, limit: int = 20) -> list[dict]:
    by_id = {it.id: it for it in run.items}
    reasons: dict[str, dict] = {}
    for v in run.verdicts:
        if v["verdict"] is None:
            continue
        reasons.setdefault(v["item_id"], {}).setdefault(str(v["verdict"]), v["reason"])
    flipped = [it for it in st.items if it.n_valid and it.flip_rate > 0]
    flipped.sort(key=lambda it: (-it.flip_rate, abs(it.pass_share - 0.5), it.item_id))
    out = []
    for it in flipped[:limit]:
        item = by_id.get(it.item_id)
        out.append(
            {
                "id": it.item_id,
                "counts": {str(k): v for k, v in it.counts.items()},
                "n_valid": it.n_valid,
                "n_invalid": it.n_invalid,
                "flip_rate": _r(it.flip_rate),
                "input": item.input if item else "",
                "output": item.output if item else "",
                "reference": (item.reference if item else None) or "",
                "reasons": reasons.get(it.item_id, {}),
            }
        )
    return out


def to_dict(run: RunData, st: RunStats, *, unstable_limit: int = 20) -> dict:
    meta = run.meta
    totals = run.totals()
    return {
        "tool": {"name": "judgejitter", "version": __version__},
        "run": {
            "run_id": meta["run_id"],
            "name": meta.get("name"),
            "created_at": meta.get("created_at"),
            "finished_at": meta.get("finished_at"),
            "status": meta.get("status"),
            "model": meta["model"],
            "api_version": meta.get("api_version"),
            "scale": meta["scale"],
            "pass_threshold": meta.get("pass_threshold"),
            "k": meta["k"],
            "structured_output": meta.get("structured"),
            "prompt_sha": meta.get("prompt_sha"),
            "items_path": meta.get("items_path"),
            "settings": meta.get("settings"),
        },
        "totals": {
            "items": st.n_items,
            "calls": totals["calls"],
            "failed_calls": totals["errors"],
            "input_tokens": totals["input_tokens"],
            "output_tokens": totals["output_tokens"],
            "cost_usd": _r(totals["cost_usd"], 6),
            "cost_known_for_every_call": totals["cost_known"],
        },
        "agreement": {
            "fleiss_kappa": _r(st.kappa),
            "icc_single": _r(st.icc.single) if st.icc else None,
            "icc_average": _r(st.icc.average) if st.icc else None,
            "items_all_agree": _r(st.exact_agreement),
            "items_flipped": st.items_flipped,
            "mean_flip_rate": _r(st.mean_flip_rate),
            "flip_histogram": flip_histogram(st),
        },
        "pass_rate": {
            "per_repeat": [_r(x) for x in st.per_repeat_pass_rates],
            "single_run": _iv(st.single),
            "majority_of_k": _iv(st.majority_k),
            "single_run_with_item_sampling": _iv(st.single_full),
            "majority_of_k_with_item_sampling": _iv(st.majority_k_full),
            "target_width_points": st.target_points,
            "recommended_k": st.recommended_k,
            "by_k": [_iv(iv) for iv in st.curve],
        },
        "unstable_items": unstable_items(run, st, unstable_limit),
    }


def flip_histogram(st: RunStats) -> list[dict]:
    """Items by how many of their verdicts disagree with the item's majority."""
    counts = Counter()
    for it in st.items:
        if it.n_valid:
            top = max(it.counts.values()) if it.counts else 0
            counts[it.n_valid - top] += 1
    top_bin = max([st.k // 2 if st.scale == "pass-fail" else st.k - 1] + list(counts))
    return [{"disagree": d, "items": counts.get(d, 0)} for d in range(0, top_bin + 1)]


# ---------------------------------------------------------------------------
# SVG charts (no scripts needed to draw them; a small script adds tooltips)


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _nice_step(span: float, target: int = 5) -> float:
    raw = span / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def svg_histogram(hist: list[dict], k: int, n_items: int) -> str:
    w, h = 560, 260
    left, right, top, bottom = 48, 16, 28, 56
    pw, ph = w - left - right, h - top - bottom
    ymax = max([b["items"] for b in hist] + [1])
    step = _nice_step(ymax, 4)
    ytop = math.ceil(ymax / step) * step
    n = len(hist)
    band = pw / n
    bar_w = min(band - 2, 96)
    parts = [
        f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-labelledby="hist-title hist-desc">',
        '<title id="hist-title">Flip rate per item</title>',
        f'<desc id="hist-desc">Number of items by how many of their {k} verdicts disagree with the majority.</desc>',
    ]
    tick = 0.0
    while tick <= ytop + 1e-9:
        y = top + ph - tick / ytop * ph
        parts.append(f'<line class="grid" x1="{left}" x2="{w - right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:g}</text>')
        tick += step
    for i, b in enumerate(hist):
        cx = left + band * (i + 0.5)
        bh = 0 if ytop == 0 else b["items"] / ytop * ph
        x0 = cx - bar_w / 2
        y0 = top + ph - bh
        share = b["items"] / n_items * 100 if n_items else 0
        rate = b["disagree"] / k * 100 if k else 0
        tip = f"{b['items']} items ({share:.1f}%) with {b['disagree']} of {k} verdicts against the majority"
        r = min(4, bh / 2, bar_w / 2)
        if bh > 0:
            path = (
                f"M{x0:.1f},{top + ph:.1f} L{x0:.1f},{y0 + r:.1f} Q{x0:.1f},{y0:.1f} {x0 + r:.1f},{y0:.1f} "
                f"L{x0 + bar_w - r:.1f},{y0:.1f} Q{x0 + bar_w:.1f},{y0:.1f} {x0 + bar_w:.1f},{y0 + r:.1f} "
                f"L{x0 + bar_w:.1f},{top + ph:.1f} Z"
            )
            parts.append(f'<path class="bar" d="{path}"/>')
        parts.append(
            f'<rect class="hit" x="{cx - band / 2:.1f}" y="{top}" width="{band:.1f}" height="{ph}" data-tip="{_esc(tip)}"/>'
        )
        parts.append(f'<text class="value" x="{cx:.1f}" y="{y0 - 8:.1f}" text-anchor="middle">{b["items"]}</text>')
        parts.append(f'<text class="tick" x="{cx:.1f}" y="{top + ph + 20:.1f}" text-anchor="middle">{rate:.0f}%</text>')
        parts.append(
            f'<text class="tick muted" x="{cx:.1f}" y="{top + ph + 36:.1f}" text-anchor="middle">{b["disagree"]} of {k}</text>'
        )
    parts.append(f'<line class="axis" x1="{left}" x2="{w - right}" y1="{top + ph}" y2="{top + ph}"/>')
    parts.append(f'<text class="tick" x="0" y="{top - 12}" text-anchor="start">items</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_intervals(by_k: list[dict], recommended: int | None, observed_k: int) -> str:
    w, h = 560, 280
    left, right, top, bottom = 48, 16, 36, 56
    pw, ph = w - left - right, h - top - bottom
    lows = [b["low"] * 100 for b in by_k]
    highs = [b["high"] * 100 for b in by_k]
    lo, hi = min(lows), max(highs)
    span = max(hi - lo, 2.0)
    step = _nice_step(span, 4)
    ylo = math.floor((lo - span * 0.1) / step) * step
    yhi = math.ceil((hi + span * 0.1) / step) * step
    ylo, yhi = max(ylo, 0.0), min(yhi, 100.0)
    if yhi <= ylo:
        ylo, yhi = max(lo - 1, 0), min(hi + 1, 100)

    def y_of(v: float) -> float:
        return top + ph - (v - ylo) / (yhi - ylo) * ph

    n = len(by_k)
    band = pw / n
    parts = [
        f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-labelledby="iv-title iv-desc">',
        '<title id="iv-title">Pass rate on a rerun, by calls per item</title>',
        '<desc id="iv-desc">For each number of judge calls per item (majority vote), the middle 95% of '
        "pass rates you would get by rerunning on the same items.</desc>",
    ]
    tick = ylo
    while tick <= yhi + 1e-9:
        y = y_of(tick)
        parts.append(f'<line class="grid" x1="{left}" x2="{w - right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:g}%</text>')
        tick += step
    for i, b in enumerate(by_k):
        cx = left + band * (i + 0.5)
        y_lo, y_hi, y_mid = y_of(b["low"] * 100), y_of(b["high"] * 100), y_of(b["pass_rate"] * 100)
        tip = (
            f"{b['k']} call{'s' if b['k'] > 1 else ''} per item: {b['pass_rate'] * 100:.1f}% "
            f"(95% of reruns between {b['low'] * 100:.1f}% and {b['high'] * 100:.1f}%, width {b['width_points']:.1f} points)"
        )
        if y_lo - y_hi < 0.5:
            y_lo, y_hi = y_mid + 0.25, y_mid - 0.25
        parts.append(f'<line class="whisker" x1="{cx:.1f}" x2="{cx:.1f}" y1="{y_lo:.1f}" y2="{y_hi:.1f}"/>')
        for yy in (y_lo, y_hi):
            parts.append(f'<line class="whisker" x1="{cx - 5:.1f}" x2="{cx + 5:.1f}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(f'<circle class="dot" cx="{cx:.1f}" cy="{y_mid:.1f}" r="4"/>')
        parts.append(
            f'<rect class="hit" x="{cx - band / 2:.1f}" y="{top}" width="{band:.1f}" height="{ph}" data-tip="{_esc(tip)}"/>'
        )
        if b["k"] in (1, observed_k, recommended):
            parts.append(
                f'<text class="value" x="{cx:.1f}" y="{y_hi - 10:.1f}" text-anchor="middle">{b["width_points"]:.1f}</text>'
            )
        cls = "tick strong" if b["k"] == recommended else "tick"
        parts.append(f'<text class="{cls}" x="{cx:.1f}" y="{top + ph + 20:.1f}" text-anchor="middle">{b["k"]}</text>')
        note = "recommended" if b["k"] == recommended else ("you ran" if b["k"] == observed_k else "")
        if note:
            parts.append(
                f'<text class="tick muted" x="{cx:.1f}" y="{top + ph + 36:.1f}" text-anchor="middle">{note}</text>'
            )
    parts.append(f'<line class="axis" x1="{left}" x2="{w - right}" y1="{top + ph}" y2="{top + ph}"/>')
    parts.append(f'<text class="tick" x="0" y="{top - 16}" text-anchor="start">pass rate</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# HTML


def _pct(x, nd=1):
    return "n/a" if x is None else f"{x * 100:.{nd}f}%"


def _num(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _clip(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _env() -> Environment:
    env = Environment(autoescape=True, trim_blocks=True, lstrip_blocks=True)
    env.filters.update(pct=_pct, num=_num, clip=_clip)
    return env


def render_html(data: dict, *, title: str | None = None) -> str:
    template = resources.files("judgejitter").joinpath("templates/report.html.j2").read_text(encoding="utf-8")
    run = data["run"]
    pr = data["pass_rate"]
    ctx = {
        "d": data,
        "title": title or f"Judge stability: {run.get('name') or run['run_id']}",
        "hist_svg": Markup(svg_histogram(data["agreement"]["flip_histogram"], run["k"], data["totals"]["items"])),
        "iv_svg": Markup(svg_intervals(pr["by_k"], pr["recommended_k"], pr["majority_of_k"]["k"])),
        "repo_url": REPO_URL,
        "version": __version__,
        "json_blob": Markup(json.dumps(data, ensure_ascii=False).replace("</", "<\\/")),
    }
    return _env().from_string(template).render(**ctx)
