# Changelog

## 0.1.0 (2026-11-01)

First release.

- `jj run`: K judge calls per item on the Claude API, concurrent, with a cost cap, a dry run, structured or free-text verdicts, and every call stored in DuckDB.
- `jj resume`: finish a stopped run and retry failed calls.
- `jj report`: one self-contained HTML report and a JSON file: Fleiss' kappa or ICC(1,1), flip rates, a flip-rate histogram, single-run versus majority-vote intervals, the recommended calls per item, and the 20 least stable items.
- `jj compare`: paired comparison of two runs with an exact McNemar test and the list of changed items.
- Input from JSONL, CSV or a promptfoo results file.
