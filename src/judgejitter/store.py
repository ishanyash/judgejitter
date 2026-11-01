"""Every verdict goes into a DuckDB file, one row per judge call."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from .items import Item

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id VARCHAR PRIMARY KEY,
        name VARCHAR,
        created_at VARCHAR,
        finished_at VARCHAR,
        status VARCHAR,
        model VARCHAR,
        api_version VARCHAR,
        scale VARCHAR,
        pass_threshold INTEGER,
        k INTEGER,
        n_items INTEGER,
        structured BOOLEAN,
        prompt_sha VARCHAR,
        prompt_template VARCHAR,
        items_path VARCHAR,
        settings VARCHAR,
        tool_version VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS items (
        run_id VARCHAR,
        item_id VARCHAR,
        position INTEGER,
        input VARCHAR,
        output VARCHAR,
        reference VARCHAR,
        PRIMARY KEY (run_id, item_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS verdicts (
        run_id VARCHAR,
        item_id VARCHAR,
        repeat INTEGER,
        verdict VARCHAR,
        reason VARCHAR,
        raw VARCHAR,
        error VARCHAR,
        stop_reason VARCHAR,
        input_tokens INTEGER,
        output_tokens INTEGER,
        cache_write_tokens INTEGER,
        cache_read_tokens INTEGER,
        cost_usd DOUBLE,
        latency_ms INTEGER,
        model VARCHAR,
        created_at VARCHAR,
        PRIMARY KEY (run_id, item_id, repeat)
    )
    """,
]


def now(timespec: str = "seconds") -> str:
    return datetime.now(UTC).isoformat(timespec=timespec)


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


@dataclass
class Verdict:
    item_id: str
    repeat: int
    verdict: str | int | None
    reason: str | None = None
    raw: str | None = None
    error: str | None = None
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: int | None = None
    model: str | None = None


@dataclass
class RunData:
    meta: dict
    items: list[Item]
    verdicts: list[dict]

    def verdict_lists(self) -> dict[str, list]:
        """item id -> verdicts in repeat order (None for a failed call)."""
        scale = self.meta["scale"]
        k = self.meta["k"]
        out: dict[str, list] = {it.id: [None] * k for it in self.items}
        for v in self.verdicts:
            if v["item_id"] in out and 1 <= v["repeat"] <= k:
                out[v["item_id"]][v["repeat"] - 1] = _decode(v["verdict"], scale)
        return out

    def totals(self) -> dict:
        costs = [v["cost_usd"] for v in self.verdicts if v["cost_usd"] is not None]
        return {
            "calls": len(self.verdicts),
            "errors": sum(1 for v in self.verdicts if v["verdict"] is None),
            "input_tokens": sum(v["input_tokens"] or 0 for v in self.verdicts),
            "output_tokens": sum(v["output_tokens"] or 0 for v in self.verdicts),
            "cost_usd": sum(costs) if costs else None,
            "cost_known": len(costs) == len(self.verdicts),
        }


def _decode(value, scale: str):
    if value is None:
        return None
    return int(value) if scale == "1-5" else str(value)


class Store:
    def __init__(self, path: str | Path, read_only: bool = False):
        self.path = Path(path)
        if read_only and not self.path.exists():
            raise FileNotFoundError(f"{self.path}: no such database (run `jj run` first)")
        self.con = duckdb.connect(str(self.path), read_only=read_only)
        if not read_only:
            for stmt in SCHEMA:
                self.con.execute(stmt)

    def close(self) -> None:
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- writing -----------------------------------------------------------

    def create_run(self, meta: dict, items: list[Item]) -> str:
        run_id = meta.get("run_id") or new_run_id()
        row = {
            "run_id": run_id,
            "name": meta.get("name"),
            "created_at": now("microseconds"),
            "finished_at": None,
            "status": "running",
            "model": meta["model"],
            "api_version": meta.get("api_version"),
            "scale": meta["scale"],
            "pass_threshold": meta.get("pass_threshold"),
            "k": meta["k"],
            "n_items": len(items),
            "structured": meta.get("structured", True),
            "prompt_sha": meta.get("prompt_sha"),
            "prompt_template": meta.get("prompt_template"),
            "items_path": meta.get("items_path"),
            "settings": json.dumps(meta.get("settings") or {}, sort_keys=True),
            "tool_version": meta.get("tool_version"),
        }
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        self.con.execute(f"INSERT INTO runs ({cols}) VALUES ({marks})", list(row.values()))
        self.con.executemany(
            "INSERT INTO items (run_id, item_id, position, input, output, reference) VALUES (?, ?, ?, ?, ?, ?)",
            [(run_id, it.id, n, it.input, it.output, it.reference) for n, it in enumerate(items)],
        )
        return run_id

    def add_verdict(self, run_id: str, v: Verdict) -> None:
        verdict = None if v.verdict is None else str(v.verdict)
        self.con.execute(
            """
            INSERT INTO verdicts (run_id, item_id, repeat, verdict, reason, raw, error, stop_reason,
                input_tokens, output_tokens, cache_write_tokens, cache_read_tokens, cost_usd,
                latency_ms, model, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id, v.item_id, v.repeat, verdict, v.reason, v.raw, v.error, v.stop_reason,
                v.input_tokens, v.output_tokens, v.cache_write_tokens, v.cache_read_tokens,
                v.cost_usd, v.latency_ms, v.model, now(),
            ],
        )  # fmt: skip

    def delete_failed(self, run_id: str) -> int:
        n = self.con.execute("SELECT count(*) FROM verdicts WHERE run_id = ? AND verdict IS NULL", [run_id]).fetchone()[
            0
        ]
        self.con.execute("DELETE FROM verdicts WHERE run_id = ? AND verdict IS NULL", [run_id])
        return int(n)

    def finish_run(self, run_id: str, status: str) -> None:
        self.con.execute("UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?", [status, now(), run_id])

    # -- reading -----------------------------------------------------------

    def done_pairs(self, run_id: str) -> set[tuple[str, int]]:
        rows = self.con.execute("SELECT item_id, repeat FROM verdicts WHERE run_id = ?", [run_id]).fetchall()
        return {(r[0], int(r[1])) for r in rows}

    def runs(self) -> list[dict]:
        cur = self.con.execute(
            """
            SELECT r.run_id, r.name, r.created_at, r.status, r.model, r.api_version, r.scale, r.k, r.n_items,
                   count(v.repeat) AS calls, sum(v.cost_usd) AS cost_usd
            FROM runs r LEFT JOIN verdicts v ON v.run_id = r.run_id
            GROUP BY r.run_id, r.name, r.created_at, r.status, r.model, r.api_version, r.scale, r.k, r.n_items
            ORDER BY r.created_at, r.run_id
            """
        )
        return _dicts(cur)

    def resolve(self, ref: str | None) -> str:
        """Find a run by id, id prefix, name, or 'latest' (the default)."""
        runs = self.runs()
        if not runs:
            raise LookupError("no runs in this database")
        if ref in (None, "", "latest"):
            return runs[-1]["run_id"]
        exact = [r for r in runs if r["run_id"] == ref or r["name"] == ref]
        if exact:
            return exact[-1]["run_id"]
        prefix = [r for r in runs if r["run_id"].startswith(ref)]
        if len(prefix) == 1:
            return prefix[0]["run_id"]
        if len(prefix) > 1:
            raise LookupError(f"{ref!r} matches more than one run: " + ", ".join(r["run_id"] for r in prefix))
        raise LookupError(f"no run called {ref!r}; `jj runs` lists them")

    def load(self, run_id: str) -> RunData:
        meta = _dicts(self.con.execute("SELECT * FROM runs WHERE run_id = ?", [run_id]))
        if not meta:
            raise LookupError(f"no run {run_id!r}")
        meta = meta[0]
        meta["settings"] = json.loads(meta["settings"] or "{}")
        meta["structured"] = bool(meta["structured"])
        items = [
            Item(r["item_id"], r["input"], r["output"], r["reference"])
            for r in _dicts(self.con.execute("SELECT * FROM items WHERE run_id = ? ORDER BY position", [run_id]))
        ]
        verdicts = _dicts(
            self.con.execute("SELECT * FROM verdicts WHERE run_id = ? ORDER BY item_id, repeat", [run_id])
        )
        return RunData(meta, items, verdicts)


def _dicts(cur) -> list[dict]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
