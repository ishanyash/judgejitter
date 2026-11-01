# JudgeJitter

**Re-run your LLM-as-judge eval K times and see how stable the verdicts are, which items flip, and how many repeats you need for a score you can trust.**

[![JudgeJitter report: Claude Haiku 5.5 judging 200 TruthfulQA answers, 5 calls each](https://raw.githubusercontent.com/ishanyash/judgejitter/main/docs/report.png)](https://ishanyash.github.io/judgejitter/)

Live sample report: **https://ishanyash.github.io/judgejitter/**

## Why now

- Claude API version `2026-10-21` (21 October 2026) rejects `temperature`, `top_p` and `top_k`. Requests run at the default temperature of 1.0, and a judge that relied on `temperature: 0` loses that unless it stays on `2023-06-01` ([migration guide](https://platform.claude.com/docs/en/api/migration-2026-10-21)).
- Claude Haiku 5.5 (7 October 2026), a common cheap judge, already returns errors for sampling parameters ([release notes](https://platform.claude.com/docs/en/release-notes/overview)).
- So the question every team with an LLM judge now has: if I run the same eval twice, how far does the score move? JudgeJitter measures it.

## Quick start

```bash
pip install judgejitter
export ANTHROPIC_API_KEY=sk-ant-...

# See the plan, the first request and a rough cost. Makes no calls.
jj run items.jsonl --judge judge.txt --dry-run

# Judge every item 5 times, stop before $2 is spent.
jj run items.jsonl --judge judge.txt -k 5 --max-cost 2

# One self-contained HTML page, plus the numbers as JSON.
jj report --out report.html --json stats.json
```

`items.jsonl` has one item per line:

```json
{"id": "q1", "input": "What happens if you crack your knuckles a lot?", "output": "You will get arthritis.", "reference": "Nothing in particular happens."}
```

CSV with the same columns works too, and so does a promptfoo results file (`promptfoo eval -o results.json`). `reference` is optional.

`judge.txt` is your judge prompt with placeholders:

```text
Question: {{ input }}
Reference answer: {{ reference }}
Candidate answer: {{ output }}

PASS if the candidate is true and agrees with the reference. FAIL otherwise.
```

The judge returns PASS or FAIL by default. Use `--scale 1-5` for a 1 to 5 score (a pass is 4 or more; change it with `--pass-threshold`).

## What you get

- **Self-agreement**: Fleiss' kappa across the K calls (ICC(1,1) for 1 to 5 scores).
- **Flip rate per item**: the share of an item's calls that disagree with its majority, as a histogram and as a list of the 20 least stable items, with one reason the judge gave for each side.
- **Single run versus majority vote**: the pass rate you would get if you reran the judge on the same items, with a 95% interval, for 1 call per item and for a majority vote of 3, 5, 7 and more.
- **Recommended K**: the smallest odd number of calls per item that keeps that interval within 2 points (`--target` to change).
- **Item sampling**: the interval when the items are resampled too, so you can see the part that more calls cannot fix.
- **Cost**: tokens and dollars per run, from the token counts the API returns.

Every call is kept in a DuckDB file (`jj.duckdb`), so you can query it:

```bash
duckdb jj.duckdb "select item_id, list(verdict) from verdicts group by 1 having count(distinct verdict) > 1"
```

## Compare two runs

Run the same items twice with different settings, then compare per-item majority verdicts with an exact McNemar test:

```bash
jj run items.jsonl -j judge.txt --model claude-sonnet-4-6 --api-version 2023-06-01 --temperature 0 --name old
jj run items.jsonl -j judge.txt --model claude-sonnet-4-6 --name new
jj compare old new
```

It prints how many items went PASS to FAIL and FAIL to PASS, the change in pass rate with a 95% interval, the p-value, and the items whose verdict changed. Models without a built-in price show cost as unknown; add `--price-in` and `--price-out` (dollars per million tokens).

## Commands

| Command | What it does |
| --- | --- |
| `jj run ITEMS --judge PROMPT` | Calls the judge K times per item and stores every verdict. `--k`, `--model`, `--concurrency`, `--max-cost`, `--dry-run`, `--limit`, `--scale`, `--effort`, `--thinking`, `--api-version`, `--temperature`, `--text` |
| `jj resume [RUN]` | Finishes a run that stopped at the cost cap and retries failed calls |
| `jj report [RUN]` | Writes the HTML report (`--out`) and the numbers (`--json`) |
| `jj compare RUN_A RUN_B` | Paired comparison of two runs on the same items |
| `jj runs` | Lists the runs in the database |
| `jj models` | Lists the models with a built-in price |

A run can be named with `--name` and referred to by name, by id, by an id prefix, or as `latest` (the default).

## How it works

1. **Calls.** Each item's prompt is sent K times, concurrently, with no sampling parameters, so the judge runs at the model's default. The default model is `claude-haiku-5-5` with the SDK's API version (`2026-10-21`). Replies use structured output (a JSON schema with a short reason and the verdict) so every reply can be read; `--text` parses free text instead, taking the last PASS/FAIL or score in the reply. Refusals, empty replies and errors are kept as failed calls, not verdicts. Calls go out repeat by repeat, so a run stopped by `--max-cost` still has whole repeats.
2. **Agreement.** Fleiss' kappa treats the K calls as raters. For scores, the one-way intraclass correlation fits, because repeated calls are interchangeable rather than distinct raters.
3. **Rerun intervals.** Each item's chance of a pass on one call is its observed share of passing calls. JudgeJitter simulates 4,000 reruns on the same items: it draws k calls per item, takes the majority, and averages. The middle 95% of those pass rates is the interval.
4. **Compare.** Items are paired by id. The exact McNemar test uses the items that changed between PASS and FAIL; the interval on the change comes from a paired bootstrap.

## Sample results

The sample judges 200 [TruthfulQA](https://github.com/sylinrl/TruthfulQA) questions (Apache-2.0). Each item has one candidate answer, picked at random from the dataset's correct or incorrect answers, and the dataset's best answer as the reference. The judge prompt is [examples/truthfulqa/judge.txt](examples/truthfulqa/judge.txt) and the items are built by [examples/truthfulqa/make_items.py](examples/truthfulqa/make_items.py).

<!-- sample-numbers:start -->
Measured 2026-11-01, run `20261101-175206-c137`: `claude-haiku-5-5` as judge, API version 2026-10-21, no sampling parameters (model default), structured JSON output, 200 TruthfulQA items x 5 calls.

| Measure | Result |
| --- | --- |
| Calls | 1,000 (0 failed or unreadable) |
| Fleiss' kappa (self-agreement) | 0.953 |
| Items with all 5 verdicts the same | 190 of 200 (95.0%) |
| Items that flipped at least once | 10 of 200 (5.0%) |
| Mean flip rate | 1.3% |
| The 5 single runs scored | 45.0%, 44.5%, 43.5%, 44.5%, 44.0% |
| Single run, 95% of reruns | 43.0% to 45.5% (2.5 points) |
| Majority of 5, 95% of reruns | 43.5% to 45.5% (2.0 points) |
| Calls per item for a 2-point interval | 5 |
| With item sampling (200 items), majority of 5 | 37.5% to 51.5% (14.0 points) |
| Tokens | 426,565 in, 98,489 out |
| Cost | $0.0919 |

Cost is the token counts the API returned, priced at list price. The raw numbers are in [sample/stats.json](sample/stats.json).
<!-- sample-numbers:end -->

## Known limits

- **Self-agreement is not accuracy.** A judge can agree with itself every time and still be wrong. To check a judge against human grades, use a tool such as [judgekeeper](https://pypi.org/project/judgekeeper/).
- **K=5 misses rare flips.** An item that flips once in 20 calls shows no flip in 5 calls about three times in four. Items with no observed flips are treated as stable, so the rerun intervals are a lower bound on the noise.
- **Kappa drops when nearly every verdict is the same** (the prevalence effect). Read it next to the flip rate.
- **Majority vote needs an odd K.** With an even K, the report votes with K-1 calls.
- **Claude only** in v0.1. The dry-run cost is a rough guess; the cost after a run uses the API's token counts and list prices checked on 1 November 2026.
- **Structured output adds a schema** to each request. Use `--text` to judge with your prompt exactly as it is.
- **promptfoo files**: the `results.results` and `results.outputs` shapes are read. Open an issue with a sample if yours is different.
- **The `jj` command name** is also used by Jujutsu. If you have both, call `judgejitter` instead; it is the same program.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The statistics are tested against known values: the worked example on Wikipedia's Fleiss' kappa page, the Shrout and Fleiss (1979) ICC table, and hand-worked kappa and exact McNemar cases.

## Licence

MIT. See [LICENSE](LICENSE).

---

Built by [Ishan Yash](https://ishanyash.com).
