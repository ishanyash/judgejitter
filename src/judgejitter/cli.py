"""The jj command line."""

from __future__ import annotations

import json
import math
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from . import __version__
from .compare import compare_runs
from .items import ItemsError, load_items
from .pricing import PRICES, Price, price_for
from .prompt import JudgePrompt, PromptError
from .report import analyse, render_html, to_dict
from .runner import DEFAULT_MODEL, FatalRunError, RunConfig, build_request, estimate, execute, start_run
from .store import Store

app = typer.Typer(
    help="Re-run your LLM-as-judge eval K times and see how stable the verdicts are.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

DB_HELP = "DuckDB file that holds every run."


class Scale(str, Enum):
    pass_fail = "pass-fail"
    one_to_five = "1-5"


class Effort(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    xhigh = "xhigh"
    max = "max"


class Thinking(str, Enum):
    adaptive = "adaptive"
    disabled = "disabled"


def _fail(msg: str) -> None:
    typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"judgejitter {__version__}")
        raise typer.Exit()


@app.callback()
def main_options(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version and exit.")
    ] = False,
) -> None:
    pass


def _fmt_pct(x: float | None, nd: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{nd}f}%"


def _progress_printer(total: int):
    tty = sys.stderr.isatty()
    step = max(total // 10, 1)

    def show(p) -> None:
        if tty:
            sys.stderr.write(f"\r  {p.done}/{p.total} calls  {p.errors} failed  ${p.cost_usd:.4f}   ")
            sys.stderr.flush()
        elif p.done % step == 0 or p.done == p.total:
            sys.stderr.write(f"  {p.done}/{p.total} calls  {p.errors} failed  ${p.cost_usd:.4f}\n")

    return show, tty


def _print_summary(store: Store, run_id: str, target: float) -> None:
    run_data = store.load(run_id)
    data = to_dict(run_data, analyse(run_data, target_points=target))
    tot, ag, pr = data["totals"], data["agreement"], data["pass_rate"]
    typer.echo("")
    typer.echo(f"run {run_id} ({data['run']['status']})")
    cost = "unknown" if tot["cost_usd"] is None else f"${tot['cost_usd']:.4f}"
    typer.echo(f"  calls           {tot['calls']} ({tot['failed_calls']} failed), cost {cost}")
    if ag["fleiss_kappa"] is not None:
        typer.echo(f"  Fleiss kappa    {ag['fleiss_kappa']:.3f}")
    if ag["icc_single"] is not None:
        typer.echo(f"  ICC(1,1)        {ag['icc_single']:.3f}")
    typer.echo(
        f"  items flipped   {ag['items_flipped']} of {tot['items']} ({_fmt_pct(ag['items_flipped'] / max(tot['items'], 1))})"
    )
    s, m = pr["single_run"], pr["majority_of_k"]
    typer.echo(
        f"  single run      {_fmt_pct(s['pass_rate'])}  (95% of reruns {_fmt_pct(s['low'])} to {_fmt_pct(s['high'])}, {s['width_points']:.1f} points)"
    )
    typer.echo(
        f"  majority of {m['k']:<3} {_fmt_pct(m['pass_rate'])}  (95% of reruns {_fmt_pct(m['low'])} to {_fmt_pct(m['high'])}, {m['width_points']:.1f} points)"
    )
    rec = pr["recommended_k"]
    typer.echo(f"  calls per item for a {target:g}-point interval: {rec if rec else 'more than 51'}")
    typer.echo(f"\nNext: jj report {run_id}")


@app.command()
def run(
    items: Annotated[Path, typer.Argument(help="Items to judge: .jsonl, .csv, or a promptfoo results .json.")],
    judge: Annotated[
        Path, typer.Option("--judge", "-j", help="Judge prompt file with {{ input }}, {{ output }}, {{ reference }}.")
    ],
    k: Annotated[int, typer.Option("--k", "-k", min=2, help="Judge calls per item.")] = 5,
    model: Annotated[str, typer.Option(help="Claude model id.")] = DEFAULT_MODEL,
    scale: Annotated[Scale, typer.Option(help="What the judge returns.")] = Scale.pass_fail,
    pass_threshold: Annotated[int, typer.Option(min=1, max=5, help="With --scale 1-5, the lowest passing score.")] = 4,
    concurrency: Annotated[int, typer.Option(min=1, max=64, help="Calls in flight at once.")] = 8,
    max_cost: Annotated[
        float | None, typer.Option(help="Stop starting new calls once spend would pass this many US dollars.")
    ] = None,
    db: Annotated[Path, typer.Option(help=DB_HELP)] = Path("jj.duckdb"),
    name: Annotated[str | None, typer.Option(help="A name for this run, used in reports and `jj compare`.")] = None,
    fmt: Annotated[str, typer.Option("--format", help="auto, jsonl, csv or promptfoo.")] = "auto",
    limit: Annotated[int | None, typer.Option(min=1, help="Only judge the first N items.")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the plan and a cost estimate. Makes no calls.")
    ] = False,
    structured: Annotated[
        bool,
        typer.Option("--structured/--text", help="JSON output with a fixed schema (default), or parse free text."),
    ] = True,
    effort: Annotated[Effort | None, typer.Option(help="output_config.effort. Default: the model's own.")] = None,
    thinking: Annotated[Thinking | None, typer.Option(help="Thinking mode. Default: the model's own.")] = None,
    max_tokens: Annotated[int, typer.Option(min=16, help="max_tokens per call (thinking counts toward it).")] = 4096,
    api_version: Annotated[
        str | None, typer.Option(help="anthropic-version header. Default: the SDK's (2026-10-21).")
    ] = None,
    temperature: Annotated[
        float | None, typer.Option(help="Only with --api-version 2023-06-01 on a model that accepts it.")
    ] = None,
    price_in: Annotated[float | None, typer.Option(help="US dollars per million input tokens.")] = None,
    price_out: Annotated[float | None, typer.Option(help="US dollars per million output tokens.")] = None,
    target: Annotated[float, typer.Option(help="Interval width, in points, for the recommended calls per item.")] = 2.0,
) -> None:
    """Call the judge K times per item and store every verdict."""
    try:
        item_list = load_items(items, fmt)
        prompt = JudgePrompt.from_text(judge.read_text(encoding="utf-8"))
        prompt.render(item_list[0])
    except (ItemsError, PromptError, OSError) as e:
        _fail(str(e))
    if limit:
        item_list = item_list[:limit]
    if temperature is not None and api_version != "2023-06-01":
        _fail(
            "API version 2026-10-21 rejects temperature, top_p and top_k. "
            "Add --api-version 2023-06-01 to send it, on a model that still accepts it."
        )
    if (price_in is None) != (price_out is None):
        _fail("give both --price-in and --price-out")
    price = price_for(model, price_in, price_out)
    if price is None and max_cost is not None:
        _fail(f"no price known for {model}; add --price-in and --price-out to use --max-cost")

    cfg = RunConfig(
        model=model,
        k=k,
        scale=scale.value,
        pass_threshold=pass_threshold,
        concurrency=concurrency,
        max_cost=max_cost,
        max_tokens=max_tokens,
        structured=structured,
        effort=effort.value if effort else None,
        thinking=thinking.value if thinking else None,
        api_version=api_version,
        temperature=temperature,
        price=price,
    )
    est = estimate(cfg, prompt, item_list)

    if dry_run:
        typer.echo("Plan (dry run: no calls made, nothing stored)")
        typer.echo(f"  items      {len(item_list)} from {items}")
        typer.echo(f"  calls      {est.calls:,} ({k} per item)")
        sampling = f"temperature {temperature}" if temperature is not None else "no sampling parameters (model default)"
        typer.echo(f"  model      {model}, API {api_version or 'SDK default (2026-10-21)'}, {sampling}")
        cost = "unknown (no price for this model)" if est.cost_usd is None else f"~${est.cost_usd:.4f}"
        typer.echo(
            f"  estimate   ~{est.input_tokens:,} input and ~{est.output_tokens:,} output tokens, {cost} "
            "(rough: thinking and reply length vary)"
        )
        if max_cost is not None:
            typer.echo(f"  cost cap   ${max_cost:.2f}")
        body = build_request(cfg, prompt.render(item_list[0]))
        content = body["messages"][0]["content"]
        if len(content) > 3000:
            body["messages"][0]["content"] = content[:3000] + f"... [{len(content) - 3000} more characters]"
        typer.echo("\nFirst request body:")
        typer.echo(json.dumps(body, indent=2, ensure_ascii=False))
        return

    if price is None:
        typer.secho(f"note: no price known for {model}, so cost will show as unknown", err=True)
    with Store(db) as store:
        run_id = start_run(store, item_list, prompt, cfg, name=name, items_path=str(items))
        typer.echo(f"run {run_id}: {est.calls:,} calls on {model} ({len(item_list)} items x {k})")
        show, tty = _progress_printer(est.calls)
        try:
            prog = execute(store, run_id, item_list, prompt, cfg, on_progress=show)
        except FatalRunError as e:
            store.finish_run(run_id, "failed")
            _fail(str(e))
        if tty:
            sys.stderr.write("\n")
        if prog.fatal:
            _fail(f"the first calls failed, so the run stopped: {prog.fatal}")
        if prog.stopped:
            typer.secho(f"{prog.stopped}; `jj resume {run_id} --max-cost N` continues it", err=True)
        if prog.errors:
            typer.secho(f"{prog.errors} calls failed or were unreadable, first: {prog.first_error}", err=True)
        _print_summary(store, run_id, target)


@app.command()
def resume(
    run_ref: Annotated[
        str | None, typer.Argument(metavar="RUN", help="Run id, name, or prefix. Default: latest.")
    ] = None,
    db: Annotated[Path, typer.Option(help=DB_HELP)] = Path("jj.duckdb"),
    max_cost: Annotated[
        float | None, typer.Option(help="A new cost cap for the rest of the run, in US dollars.")
    ] = None,
    concurrency: Annotated[int | None, typer.Option(min=1, max=64, help="Calls in flight at once.")] = None,
    target: Annotated[float, typer.Option(help="Interval width, in points, for the recommended calls per item.")] = 2.0,
) -> None:
    """Finish a run that stopped early, and retry its failed calls."""
    with Store(db) as store:
        try:
            run_id = store.resolve(run_ref)
        except LookupError as e:
            _fail(str(e))
        data = store.load(run_id)
        s = dict(data.meta["settings"])
        price_pair = s.pop("price_per_mtok", None)
        price = price_for(s["model"]) or (Price(*price_pair) if price_pair else None)
        if max_cost is not None:
            s["max_cost"] = max_cost
        if concurrency is not None:
            s["concurrency"] = concurrency
        cfg = RunConfig(**{**s, "price": price})
        prompt = JudgePrompt.from_text(data.meta["prompt_template"])
        missing = cfg.k * len(data.items) - len(store.done_pairs(run_id)) + data.totals()["errors"]
        typer.echo(f"run {run_id}: {missing} calls to make")
        show, tty = _progress_printer(missing)
        try:
            prog = execute(store, run_id, data.items, prompt, cfg, on_progress=show)
        except FatalRunError as e:
            _fail(str(e))
        if tty:
            sys.stderr.write("\n")
        if prog.fatal:
            _fail(f"the calls failed: {prog.fatal}")
        if prog.stopped:
            typer.secho(prog.stopped, err=True)
        _print_summary(store, run_id, target)


@app.command()
def report(
    run_ref: Annotated[
        str | None, typer.Argument(metavar="RUN", help="Run id, name, or prefix. Default: latest.")
    ] = None,
    db: Annotated[Path, typer.Option(help=DB_HELP)] = Path("jj.duckdb"),
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to write the HTML report.")] = Path("jj-report.html"),
    json_out: Annotated[Path | None, typer.Option("--json", help="Also write the numbers as JSON.")] = None,
    title: Annotated[str | None, typer.Option(help="Report title.")] = None,
    target: Annotated[float, typer.Option(help="Interval width, in points, for the recommended calls per item.")] = 2.0,
    top: Annotated[int, typer.Option(min=0, help="How many unstable items to list.")] = 20,
) -> None:
    """Write one self-contained HTML report for a run."""
    try:
        store = Store(db, read_only=True)
    except Exception as e:  # noqa: BLE001
        _fail(str(e))
    with store:
        try:
            run_id = store.resolve(run_ref)
        except LookupError as e:
            _fail(str(e))
        data = store.load(run_id)
    if not data.verdicts:
        _fail(f"run {run_id} has no verdicts yet")
    numbers = to_dict(data, analyse(data, target_points=target), unstable_limit=top)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(numbers, title=title), encoding="utf-8")
    typer.echo(f"wrote {out}")
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(numbers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        typer.echo(f"wrote {json_out}")


@app.command()
def compare(
    run_a: Annotated[str, typer.Argument(metavar="RUN_A", help="The baseline run.")],
    run_b: Annotated[str, typer.Argument(metavar="RUN_B", help="The run to compare with it.")],
    db: Annotated[Path, typer.Option(help=DB_HELP)] = Path("jj.duckdb"),
    db_b: Annotated[Path | None, typer.Option(help="DuckDB file for RUN_B, if it is in another file.")] = None,
    show: Annotated[int, typer.Option(min=0, help="How many changed items to list.")] = 20,
    json_out: Annotated[Path | None, typer.Option("--json", help="Also write the comparison as JSON.")] = None,
) -> None:
    """Compare two runs on the same items with a paired test on majority verdicts."""
    try:
        with Store(db, read_only=True) as s:
            a = s.load(s.resolve(run_a))
            if db_b is None:
                b = s.load(s.resolve(run_b))
        if db_b is not None:
            with Store(db_b, read_only=True) as s2:
                b = s2.load(s2.resolve(run_b))
    except (LookupError, FileNotFoundError) as e:
        _fail(str(e))
    res = compare_runs(a, b)
    if res["items_compared"] == 0:
        _fail("the two runs share no items with a clear majority verdict")

    def label(side: dict) -> str:
        return side["name"] or side["run_id"]

    def k_or_icc(side: dict) -> str:
        if side["fleiss_kappa"] is not None:
            return f"kappa {side['fleiss_kappa']:.3f}"
        if side["icc_single"] is not None:
            return f"ICC {side['icc_single']:.3f}"
        return "kappa n/a"

    for tag, side in (("A", res["a"]), ("B", res["b"])):
        typer.echo(
            f"{tag}  {label(side)}: {side['model']}, API {side['api_version']}, {side['k']} calls/item, "
            f"{k_or_icc(side)}, {side['items_flipped']} items flipped, "
            f"majority pass rate {_fmt_pct(side['majority_pass_rate'])}"
        )
    typer.echo("")
    typer.echo(
        f"items compared        {res['items_compared']} (skipped {len(res['items_tied_skipped'])} with a tied vote)"
    )
    typer.echo(f"PASS in A, FAIL in B  {res['pass_to_fail']}")
    typer.echo(f"FAIL in A, PASS in B  {res['fail_to_pass']}")
    typer.echo(
        f"pass rate change      {res['pass_rate_diff'] * 100:+.1f} points "
        f"(95% CI {res['pass_rate_diff_low'] * 100:+.1f} to {res['pass_rate_diff_high'] * 100:+.1f})"
    )
    typer.echo(f"McNemar exact p       {res['mcnemar_p']:.4g}")
    if res["only_in_a"] or res["only_in_b"]:
        typer.echo(f"items in only one run {len(res['only_in_a'])} in A, {len(res['only_in_b'])} in B")
    if res["changed"] and show:
        typer.echo(f"\nItems whose majority verdict changed ({len(res['changed'])}):")
        for c in res["changed"][:show]:
            ca = " ".join(f"{k}x{v}" for k, v in c["a_counts"].items())
            cb = " ".join(f"{k}x{v}" for k, v in c["b_counts"].items())
            typer.echo(f"  {c['id']:<16} {c['a']} -> {c['b']}   A: {ca}   B: {cb}")
        if len(res["changed"]) > show:
            typer.echo(f"  ... and {len(res['changed']) - show} more (--show to list more)")
    if json_out:
        json_out.write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
        typer.echo(f"\nwrote {json_out}")


@app.command()
def runs(db: Annotated[Path, typer.Option(help=DB_HELP)] = Path("jj.duckdb")) -> None:
    """List the runs in a database."""
    try:
        store = Store(db, read_only=True)
    except Exception as e:  # noqa: BLE001
        _fail(str(e))
    with store:
        rows = store.runs()
    if not rows:
        typer.echo("no runs yet")
        return
    for r in rows:
        cost = "" if r["cost_usd"] is None else f"  ${r['cost_usd']:.4f}"
        typer.echo(
            f"{r['run_id']}  {(r['name'] or '-'):<16} {r['model']:<20} {r['n_items']} items x {r['k']}  "
            f"{r['calls']} calls  {r['status']}{cost}"
        )


@app.command()
def models() -> None:
    """Show the models with a known price."""
    for model, p in PRICES.items():
        typer.echo(f"{model:<20} ${p.input:g} in / ${p.output:g} out per million tokens")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
