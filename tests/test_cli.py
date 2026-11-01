import json

import pytest

typer_testing = pytest.importorskip("typer.testing")

from conftest import PROMPT  # noqa: E402

from judgejitter.cli import app  # noqa: E402

runner = typer_testing.CliRunner()


@pytest.fixture
def files(tmp_path):
    items = tmp_path / "items.jsonl"
    items.write_text(
        "\n".join(
            json.dumps({"id": f"q{i}", "input": f"question {i}", "output": f"a{i}", "reference": "r"}) for i in range(4)
        )
    )
    judge = tmp_path / "judge.txt"
    judge.write_text(PROMPT)
    return items, judge


def test_version():
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0 and "judgejitter" in res.output


def test_dry_run_makes_no_calls(files, tmp_path, monkeypatch):
    items, judge = files
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    db = tmp_path / "jj.duckdb"
    res = runner.invoke(app, ["run", str(items), "--judge", str(judge), "--dry-run", "--db", str(db)])
    assert res.exit_code == 0, res.output
    assert "20 (5 per item)" in res.output
    assert '"model": "claude-haiku-5-5"' in res.output
    assert "temperature" not in res.output.split("First request body:")[1]
    assert not db.exists()


def test_run_without_key_fails_cleanly(files, tmp_path, monkeypatch):
    items, judge = files
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    res = runner.invoke(app, ["run", str(items), "-j", str(judge), "--db", str(tmp_path / "x.duckdb")])
    assert res.exit_code == 1 and "ANTHROPIC_API_KEY" in res.output


def test_temperature_needs_old_api_version(files):
    items, judge = files
    res = runner.invoke(app, ["run", str(items), "-j", str(judge), "--temperature", "0", "--dry-run"])
    assert res.exit_code == 1 and "2023-06-01" in res.output


def test_run_report_compare_with_fake_judge(files, tmp_path, monkeypatch):
    from conftest import FakeJudge

    import judgejitter.runner as runner_mod

    def fake_caller(cfg):
        async def close():
            return None

        return FakeJudge(seed=len(str(tmp_path))), close

    monkeypatch.setattr(runner_mod, "make_anthropic_caller", fake_caller)
    items, judge = files
    db = tmp_path / "jj.duckdb"
    for name in ("one", "two"):
        res = runner.invoke(app, ["run", str(items), "-j", str(judge), "--db", str(db), "--name", name, "-k", "3"])
        assert res.exit_code == 0, res.output
        assert "Fleiss kappa" in res.output
    out = tmp_path / "r.html"
    js = tmp_path / "r.json"
    res = runner.invoke(app, ["report", "one", "--db", str(db), "-o", str(out), "--json", str(js)])
    assert res.exit_code == 0, res.output
    assert out.read_text().startswith("<!doctype html>")
    assert json.loads(js.read_text())["run"]["name"] == "one"
    res = runner.invoke(app, ["compare", "one", "two", "--db", str(db)])
    assert res.exit_code == 0, res.output and "McNemar" in res.output
    res = runner.invoke(app, ["runs", "--db", str(db)])
    assert res.exit_code == 0 and "one" in res.output and "two" in res.output
    res = runner.invoke(app, ["resume", "two", "--db", str(db)])
    assert res.exit_code == 0, res.output
