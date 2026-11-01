"""Load eval items from JSONL, CSV or a promptfoo results file."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

INPUT_KEYS = ("input", "question", "prompt", "query")
OUTPUT_KEYS = ("output", "answer", "response", "completion", "prediction")
REFERENCE_KEYS = ("reference", "expected", "ideal", "target", "gold")


@dataclass
class Item:
    id: str
    input: str
    output: str
    reference: str | None = None


class ItemsError(ValueError):
    pass


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _pick(row: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _from_row(row: dict, n: int, where: str) -> Item:
    out = _pick(row, OUTPUT_KEYS)
    if out is None:
        raise ItemsError(f"{where}: row {n} has no output (expected one of {', '.join(OUTPUT_KEYS)})")
    ref = _pick(row, REFERENCE_KEYS)
    item_id = row.get("id")
    return Item(
        id=str(item_id) if item_id not in (None, "") else str(n),
        input=_text(_pick(row, INPUT_KEYS)),
        output=_text(out),
        reference=_text(ref) if ref is not None else None,
    )


def _check_ids(items: list[Item], where: str) -> list[Item]:
    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            raise ItemsError(f"{where}: duplicate id {it.id!r}")
        seen.add(it.id)
    if not items:
        raise ItemsError(f"{where}: no items found")
    return items


def load_jsonl(path: Path) -> list[Item]:
    items = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ItemsError(f"{path}: line {n} is not valid JSON ({e.msg})") from None
            if not isinstance(row, dict):
                raise ItemsError(f"{path}: line {n} is not a JSON object")
            items.append(_from_row(row, n, str(path)))
    return _check_ids(items, str(path))


def load_csv(path: Path) -> list[Item]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    items = [_from_row(row, n, str(path)) for n, row in enumerate(rows, start=1)]
    return _check_ids(items, str(path))


def _promptfoo_rows(data) -> list[dict]:
    """Find the per-test result rows in the shapes promptfoo has written."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    res = data.get("results")
    if isinstance(res, list):
        return res
    if isinstance(res, dict):
        for key in ("results", "outputs"):
            if isinstance(res.get(key), list):
                return res[key]
    return []


def _promptfoo_reference(row: dict) -> str | None:
    vars_ = row.get("vars") or (row.get("testCase") or {}).get("vars") or {}
    ref = _pick(vars_, REFERENCE_KEYS) if isinstance(vars_, dict) else None
    if ref is not None:
        return _text(ref)
    asserts = (row.get("testCase") or {}).get("assert") or []
    for a in asserts:
        if isinstance(a, dict) and a.get("type") in (
            "equals",
            "contains",
            "icontains",
            "similar",
            "factuality",
            "llm-rubric",
            "model-graded-closedqa",
        ):
            return _text(a.get("value"))
    return None


def load_promptfoo(path: Path) -> list[Item]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = _promptfoo_rows(data)
    if not rows:
        raise ItemsError(f"{path}: no promptfoo results found (looked in results.results and results.outputs)")
    providers = {_provider_id(r) for r in rows if isinstance(r, dict) and (r.get("response") or {}) != {}}
    items = []
    for n, row in enumerate(rows, start=1):
        prompt = row.get("prompt")
        if isinstance(prompt, dict):
            prompt = prompt.get("raw") or prompt.get("display") or prompt.get("label")
        response = row.get("response") or {}
        output = response.get("output") if isinstance(response, dict) else response
        if output is None:
            continue  # errored test with no output: nothing to judge
        test_idx, prompt_idx = row.get("testIdx"), row.get("promptIdx")
        if test_idx is not None:
            item_id = f"t{test_idx}-p{prompt_idx or 0}"
            if len(providers) > 1:
                item_id += f"-{_provider_id(row)}"
        else:
            item_id = str(row.get("id") or n)
        items.append(
            Item(
                id=item_id,
                input=_text(prompt) if prompt is not None else _text(row.get("vars")),
                output=_text(output),
                reference=_promptfoo_reference(row),
            )
        )
    return _check_ids(items, str(path))


def _provider_id(row: dict) -> str:
    p = row.get("provider")
    if isinstance(p, dict):
        return str(p.get("label") or p.get("id") or "")
    return str(p or "")


def load_items(path: str | Path, fmt: str = "auto") -> list[Item]:
    """Load items. fmt is auto, jsonl, csv or promptfoo."""
    path = Path(path)
    if not path.exists():
        raise ItemsError(f"{path}: file not found")
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix in (".jsonl", ".ndjson"):
            fmt = "jsonl"
        elif suffix == ".csv":
            fmt = "csv"
        elif suffix == ".json":
            fmt = "promptfoo"
        else:
            raise ItemsError(f"{path}: cannot tell the format from the extension; pass --format")
    loaders = {"jsonl": load_jsonl, "csv": load_csv, "promptfoo": load_promptfoo}
    if fmt not in loaders:
        raise ItemsError(f"unknown format {fmt!r}; use jsonl, csv or promptfoo")
    return loaders[fmt](path)
