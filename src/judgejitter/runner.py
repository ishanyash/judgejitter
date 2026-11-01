"""Call the judge K times per item and store every verdict."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field

from . import __version__
from .items import Item
from .pricing import Price, call_cost
from .prompt import JudgePrompt, Parsed, output_schema, parse_structured, parse_text
from .store import Store, Verdict

DEFAULT_MODEL = "claude-haiku-5-5"
# anthropic 2.x sends this anthropic-version header unless told otherwise.
SDK_API_VERSION = "2026-10-21"

# Rough numbers for --dry-run and for the cost cap before the first reply.
CHARS_PER_TOKEN = 3.2
SCHEMA_OVERHEAD_TOKENS = 300
EST_OUTPUT_TOKENS = 400


@dataclass
class RunConfig:
    model: str = DEFAULT_MODEL
    k: int = 5
    scale: str = "pass-fail"
    pass_threshold: int = 4
    concurrency: int = 8
    max_cost: float | None = None
    max_tokens: int = 4096
    structured: bool = True
    effort: str | None = None  # low, medium, high, xhigh, max; None = the model's default
    thinking: str | None = None  # adaptive or disabled; None = the model's default
    api_version: str | None = None  # None = the SDK's default (2026-10-21 for anthropic 2.x)
    temperature: float | None = None  # only accepted with api_version 2023-06-01
    timeout: float = 120.0
    max_retries: int = 4
    price: Price | None = None

    def settings(self) -> dict:
        d = asdict(self)
        d.pop("price")
        d["price_per_mtok"] = None if self.price is None else [self.price.input, self.price.output]
        return d


def build_request(cfg: RunConfig, prompt_text: str) -> dict:
    """The Messages API request body for one judge call.

    No sampling parameters are sent unless --temperature is given, so the
    judge runs at the model's default sampling.
    """
    body: dict = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "messages": [{"role": "user", "content": prompt_text}],
    }
    output_config: dict = {}
    if cfg.structured:
        output_config["format"] = {"type": "json_schema", "schema": output_schema(cfg.scale)}
    if cfg.effort:
        output_config["effort"] = cfg.effort
    if output_config:
        body["output_config"] = output_config
    if cfg.thinking:
        body["thinking"] = {"type": cfg.thinking}
    if cfg.temperature is not None:
        body["temperature"] = cfg.temperature
    return body


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 10


@dataclass
class Estimate:
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


def estimate(cfg: RunConfig, prompt: JudgePrompt, items: list[Item], calls: int | None = None) -> Estimate:
    per_item = [estimate_tokens(prompt.render(it)) + (SCHEMA_OVERHEAD_TOKENS if cfg.structured else 0) for it in items]
    n_calls = cfg.k * len(items) if calls is None else calls
    avg_in = sum(per_item) / max(len(per_item), 1)
    in_tok = int(avg_in * n_calls)
    out_tok = EST_OUTPUT_TOKENS * n_calls
    cost = call_cost(cfg.price, in_tok, out_tok)
    return Estimate(n_calls, in_tok, out_tok, cost)


# ---------------------------------------------------------------------------
# The client


class FatalRunError(RuntimeError):
    """A problem with the request itself (bad key, bad model, bad parameter)."""


Caller = Callable[[dict], Awaitable[object]]


def make_anthropic_caller(cfg: RunConfig) -> tuple[Caller, Callable[[], Awaitable[None]]]:
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise FatalRunError("the anthropic package is not installed") from e
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise FatalRunError("ANTHROPIC_API_KEY is not set (use --dry-run to check the plan without a key)")
    headers = {"anthropic-version": cfg.api_version} if cfg.api_version else None
    client = anthropic.AsyncAnthropic(max_retries=cfg.max_retries, timeout=cfg.timeout, default_headers=headers)

    async def call(body: dict):
        body = dict(body)
        extra = {}
        if "temperature" in body:  # sent raw: newer SDKs may not take it as an argument
            extra["temperature"] = body.pop("temperature")
        if extra:
            return await client.messages.create(**body, extra_body=extra)
        return await client.messages.create(**body)

    async def close():
        await client.close()

    return call, close


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _usage(resp, name: str) -> int:
    usage = getattr(resp, "usage", None)
    value = getattr(usage, name, None) if usage is not None else None
    return int(value) if isinstance(value, (int, float)) else 0


def _text_of(resp) -> str:
    parts = []
    for block in getattr(resp, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


def verdict_from_response(resp, cfg: RunConfig, item_id: str, repeat: int, latency_ms: int) -> Verdict:
    text = _text_of(resp)
    stop = getattr(resp, "stop_reason", None)
    if stop == "refusal":
        parsed = Parsed(None, None, "the model declined (stop_reason refusal)")
    elif stop == "max_tokens" and not text:
        parsed = Parsed(None, None, "hit max_tokens before any text (raise --max-tokens)")
    elif cfg.structured:
        parsed = parse_structured(text, cfg.scale)
    else:
        parsed = parse_text(text, cfg.scale)
    v = Verdict(
        item_id=item_id,
        repeat=repeat,
        verdict=parsed.verdict,
        reason=parsed.reason,
        raw=text,
        error=parsed.error,
        stop_reason=stop,
        input_tokens=_usage(resp, "input_tokens"),
        output_tokens=_usage(resp, "output_tokens"),
        cache_write_tokens=_usage(resp, "cache_creation_input_tokens"),
        cache_read_tokens=_usage(resp, "cache_read_input_tokens"),
        latency_ms=latency_ms,
        model=getattr(resp, "model", None) or cfg.model,
    )
    v.cost_usd = call_cost(cfg.price, v.input_tokens, v.output_tokens, v.cache_write_tokens, v.cache_read_tokens)
    return v


# ---------------------------------------------------------------------------
# Running


@dataclass
class Progress:
    total: int
    done: int = 0
    errors: int = 0
    cost_usd: float = 0.0
    stopped: str | None = None
    first_error: str | None = None
    fatal: str | None = None
    timings: list[int] = field(default_factory=list)


async def _run_jobs(
    jobs: list[tuple[Item, int]],
    prompt: JudgePrompt,
    cfg: RunConfig,
    caller: Caller,
    store: Store,
    run_id: str,
    on_progress: Callable[[Progress], None] | None,
) -> Progress:
    prog = Progress(total=len(jobs))
    sem = asyncio.Semaphore(cfg.concurrency)
    tasks: set[asyncio.Task] = set()
    successes = 0
    first_est = None
    if cfg.max_cost is not None and jobs:
        first_est = estimate(cfg, prompt, [jobs[0][0]], calls=1).cost_usd

    def per_call_estimate() -> float:
        if prog.done - prog.errors > 0 and prog.cost_usd > 0:
            return prog.cost_usd / max(prog.done, 1)
        return first_est or 0.0

    async def work(item: Item, repeat: int) -> None:
        nonlocal successes
        body = build_request(cfg, prompt.render(item))
        t0 = time.monotonic()
        try:
            resp = await caller(body)
        except Exception as exc:  # noqa: BLE001 - every failure is recorded, not raised
            code = _status_code(exc)
            msg = f"{type(exc).__name__}: {getattr(exc, 'message', None) or exc}"[:500]
            if code in (400, 401, 403, 404) and successes == 0:
                prog.fatal = prog.fatal or msg
            v = Verdict(item.id, repeat, None, error=msg, model=cfg.model)
        else:
            v = verdict_from_response(resp, cfg, item.id, repeat, int((time.monotonic() - t0) * 1000))
            successes += 1
        finally:
            sem.release()
        if prog.fatal and v.verdict is None and successes == 0:
            return  # do not store calls that failed for a setup reason
        store.add_verdict(run_id, v)
        prog.done += 1
        if v.verdict is None:
            prog.errors += 1
            prog.first_error = prog.first_error or v.error
        if v.cost_usd:
            prog.cost_usd += v.cost_usd
        if on_progress:
            on_progress(prog)

    for item, repeat in jobs:
        await sem.acquire()
        if prog.fatal:
            sem.release()
            break
        if cfg.max_cost is not None:
            projected = prog.cost_usd + (len(tasks) + 1) * per_call_estimate()
            if projected > cfg.max_cost:
                sem.release()
                prog.stopped = f"stopped at the cost cap of ${cfg.max_cost:.2f}"
                break
        task = asyncio.create_task(work(item, repeat))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
    if tasks:
        await asyncio.gather(*tasks)
    return prog


def plan_jobs(items: list[Item], k: int, done: set[tuple[str, int]]) -> list[tuple[Item, int]]:
    """Repeat-major order, so a run stopped early still has whole repeats."""
    return [(it, r) for r in range(1, k + 1) for it in items if (it.id, r) not in done]


def start_run(
    store: Store,
    items: list[Item],
    prompt: JudgePrompt,
    cfg: RunConfig,
    *,
    name: str | None = None,
    items_path: str | None = None,
) -> str:
    meta = {
        "name": name,
        "model": cfg.model,
        "api_version": cfg.api_version or SDK_API_VERSION,
        "scale": cfg.scale,
        "pass_threshold": cfg.pass_threshold if cfg.scale == "1-5" else None,
        "k": cfg.k,
        "structured": cfg.structured,
        "prompt_sha": prompt.sha,
        "prompt_template": prompt.template,
        "items_path": items_path,
        "settings": cfg.settings(),
        "tool_version": __version__,
    }
    return store.create_run(meta, items)


def execute(
    store: Store,
    run_id: str,
    items: list[Item],
    prompt: JudgePrompt,
    cfg: RunConfig,
    caller: Caller | None = None,
    on_progress: Callable[[Progress], None] | None = None,
) -> Progress:
    """Run every missing (item, repeat) pair of a run. Failed calls are retried."""
    store.delete_failed(run_id)
    jobs = plan_jobs(items, cfg.k, store.done_pairs(run_id))
    close = None
    if caller is None:
        caller, close = make_anthropic_caller(cfg)

    async def main() -> Progress:
        try:
            return await _run_jobs(jobs, prompt, cfg, caller, store, run_id, on_progress)
        finally:
            if close is not None:
                await close()

    prog = asyncio.run(main())
    if prog.fatal:
        store.finish_run(run_id, "failed")
    elif prog.stopped or len(store.done_pairs(run_id)) < cfg.k * len(items):
        store.finish_run(run_id, "partial")
    else:
        store.finish_run(run_id, "complete")
    return prog
