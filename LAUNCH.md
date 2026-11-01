# JudgeJitter v0.1: launch notes

Facts for posts. Every number in the results section is written by `scripts/fill_numbers.py` from the sample's `sample/stats.json`, so it matches the run exactly.

## What it does

JudgeJitter re-runs an LLM-as-judge eval K times per item at the model's default sampling and stores every verdict in DuckDB. It reports how often the judge agrees with itself, which items flip, how much a single run's pass rate moves on a rerun, and how many calls per item a majority vote needs before the pass rate holds still.

## Who it is for

Teams that grade model outputs with Claude as a judge, especially anyone whose judge used `temperature: 0` and now runs at the default because API version `2026-10-21` rejects sampling parameters.

## Why now (dated facts)

- 7 October 2026: Claude Haiku 5.5 launched; sampling parameters can return a 400 on it ([release notes](https://platform.claude.com/docs/en/release-notes/overview)).
- 21 October 2026: API version `2026-10-21` rejects `temperature`, `top_p` and `top_k`; requests use the default temperature of 1.0 ([migration guide](https://platform.claude.com/docs/en/api/migration-2026-10-21)).
- 22 October 2026: the agent-ways project reported its judge agreed with itself on over 99% of pairs at temperature 0, and that agreement at the default is unmeasured ([issue #1130](https://github.com/aaronsb/agent-ways/issues/1130)).
- Existing tools: judgekeeper measures agreement with human grades; promptfoo can repeat runs. Neither reports self-agreement across repeats, per-item flip rates, or the calls per item needed for a stable pass rate.

## The demo

Claude Haiku 5.5 (`claude-haiku-5-5`) judges 200 TruthfulQA questions (Apache-2.0), 5 calls per item, no sampling parameters, structured JSON output. Each item's candidate answer is one of the dataset's correct or incorrect answers, picked at random; the reference is the dataset's best answer.

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

## Three screenshots to take

The sample workflow saves these in `docs/`, ready to use:

1. `docs/summary.png`: the one-line finding at the top of the report (items that flipped, the single-run range, the recommended calls per item).
2. `docs/flips.png`: the flip-rate histogram.
3. `docs/repeats.png`: pass rate on a rerun for 1, 3, 5 and more calls per item, with the recommended number marked.

To take them by hand, open the live report and capture those three sections at desktop width.

## Links

- PyPI: https://pypi.org/project/judgejitter/
- Code: https://github.com/ishanyash/judgejitter
- Live sample report: https://ishanyash.github.io/judgejitter/
- Sample numbers (JSON): https://ishanyash.github.io/judgejitter/stats.json

## Install

```bash
pip install judgejitter
jj run items.jsonl --judge judge.txt --dry-run
```

## Not in v0.1

No human labelling, no providers other than Claude, no hosted service. Self-agreement is not accuracy: a judge can be stable and wrong.
