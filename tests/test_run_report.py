import json

import pytest
from conftest import PROMPT, FakeJudge, StatusError, fake_response

from judgejitter.compare import compare_runs
from judgejitter.pricing import PRICES, call_cost
from judgejitter.prompt import JudgePrompt
from judgejitter.report import analyse, render_html, to_dict
from judgejitter.runner import RunConfig, build_request, estimate, execute, plan_jobs, start_run, verdict_from_response
from judgejitter.store import Store


def make_run(tmp_path, items, k=5, seed=0, scale="pass-fail", name=None, db="jj.duckdb", **cfg_kw):
    store = Store(tmp_path / db)
    cfg = RunConfig(k=k, scale=scale, price=PRICES["claude-haiku-5-5"], concurrency=3, **cfg_kw)
    prompt = JudgePrompt.from_text(PROMPT)
    run_id = start_run(store, items, prompt, cfg, name=name)
    judge = FakeJudge(seed=seed, scale=scale)
    prog = execute(store, run_id, items, prompt, cfg, caller=judge)
    return store, run_id, prog, judge


def test_request_has_no_sampling_parameters_by_default():
    body = build_request(RunConfig(), "hello")
    assert body["model"] == "claude-haiku-5-5"
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(body)
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "effort" not in body["output_config"]
    body = build_request(RunConfig(effort="low", thinking="disabled", structured=False), "x")
    assert body["output_config"] == {"effort": "low"} and body["thinking"] == {"type": "disabled"}
    assert build_request(RunConfig(api_version="2023-06-01", temperature=0.0), "x")["temperature"] == 0.0


def test_plan_jobs_is_repeat_major_and_skips_done(items):
    jobs = plan_jobs(items[:3], 2, {("q0", 1)})
    assert [(i.id, r) for i, r in jobs] == [("q1", 1), ("q2", 1), ("q0", 2), ("q1", 2), ("q2", 2)]


def test_cost_matches_price_table():
    p = PRICES["claude-haiku-5-5"]
    assert call_cost(p, 50_000, 10_000) == pytest.approx(50_000 * 0.10 / 1e6 + 10_000 * 0.50 / 1e6)
    # Over 100k prompt tokens in one request: the higher Haiku 5.5 rates apply.
    assert call_cost(p, 200_000, 1_000) == pytest.approx(200_000 * 0.5 / 1e6 + 1_000 * 2.5 / 1e6)
    assert call_cost(None, 10, 10) is None


def test_verdict_from_response_handles_refusal_and_max_tokens():
    cfg = RunConfig(price=PRICES["claude-haiku-5-5"])
    v = verdict_from_response(fake_response("", stop="refusal"), cfg, "a", 1, 10)
    assert v.verdict is None and "declined" in v.error
    v = verdict_from_response(fake_response("", stop="max_tokens"), cfg, "a", 1, 10)
    assert "max_tokens" in v.error
    v = verdict_from_response(fake_response('{"reason": "r", "verdict": "FAIL"}'), cfg, "a", 1, 10)
    assert v.verdict == "FAIL" and v.cost_usd == pytest.approx((500 * 0.1 + 60 * 0.5) / 1e6)


def test_full_run_report_and_resume(tmp_path, items):
    store, run_id, prog, judge = make_run(tmp_path, items, name="first")
    assert prog.done == 60 and prog.errors == 0 and len(judge.calls) == 60
    data = store.load(run_id)
    assert data.meta["status"] == "complete" and data.meta["k"] == 5
    lists = data.verdict_lists()
    assert lists["q0"] == ["PASS"] * 5 and lists["q3"] == ["FAIL"] * 5
    st = analyse(data)
    assert st.n_items == 12 and 0 < st.items_flipped <= 2
    numbers = to_dict(data, st)
    assert numbers["totals"]["calls"] == 60
    assert numbers["totals"]["cost_usd"] == pytest.approx(60 * (500 * 0.1 + 60 * 0.5) / 1e6)
    assert {u["id"] for u in numbers["unstable_items"]} <= {"q1", "q2"}
    json.dumps(numbers)  # the JSON export must serialise
    page = render_html(numbers, title="Test <run>")
    assert "Test &lt;run&gt;" in page and "<svg" in page and "Fleiss" in page
    assert 'id="jj-data"' in page and "Made with" in page
    # Resume after a deleted verdict: only that call is made again.
    store.con.execute("DELETE FROM verdicts WHERE run_id = ? AND item_id = 'q5' AND repeat = 3", [run_id])
    cfg = RunConfig(k=5, price=PRICES["claude-haiku-5-5"])
    judge2 = FakeJudge()
    execute(store, run_id, items, JudgePrompt.from_text(PROMPT), cfg, caller=judge2)
    assert len(judge2.calls) == 1 and store.load(run_id).meta["status"] == "complete"
    store.close()


def test_scores_run(tmp_path, items):
    store, run_id, prog, _ = make_run(tmp_path, items, k=4, scale="1-5")
    data = store.load(run_id)
    numbers = to_dict(data, analyse(data))
    assert numbers["agreement"]["icc_single"] is not None and numbers["agreement"]["fleiss_kappa"] is None
    assert numbers["pass_rate"]["majority_of_k"]["k"] == 3  # even k is voted with k - 1
    assert "ICC" in render_html(numbers)
    store.close()


def test_cost_cap_stops_early(tmp_path, items):
    store, run_id, prog, judge = make_run(tmp_path, items, max_cost=0.0005)
    assert prog.stopped and len(judge.calls) < 60
    assert store.load(run_id).meta["status"] == "partial"
    store.close()


def test_setup_error_stops_the_run(tmp_path, items):
    store = Store(tmp_path / "jj.duckdb")
    cfg = RunConfig(k=3, concurrency=1)
    prompt = JudgePrompt.from_text(PROMPT)
    run_id = start_run(store, items, prompt, cfg)
    judge = FakeJudge(fail_first=StatusError(400, "temperature is not supported"))
    prog = execute(store, run_id, items, prompt, cfg, caller=judge)
    assert prog.fatal and "temperature" in prog.fatal
    assert len(judge.calls) <= 2
    assert store.load(run_id).meta["status"] == "failed"
    store.close()


def test_server_errors_are_recorded_not_fatal(tmp_path, items):
    store, run_id, _, _ = make_run(tmp_path, items, k=3)
    store.close()
    store = Store(tmp_path / "jj.duckdb")
    cfg = RunConfig(k=3, concurrency=2)
    prompt = JudgePrompt.from_text(PROMPT)
    run2 = start_run(store, items, prompt, cfg)

    class Flaky(FakeJudge):
        async def __call__(self, body):
            if len(self.calls) % 4 == 3:
                self.calls.append(body)
                raise StatusError(529, "overloaded")
            return await super().__call__(body)

    prog = execute(store, run2, items, prompt, cfg, caller=Flaky())
    assert prog.errors > 0 and not prog.fatal
    assert store.load(run2).totals()["errors"] == prog.errors
    store.close()


def test_compare_runs(tmp_path, items):
    store, a, _, _ = make_run(tmp_path, items, seed=1, name="a")
    store.close()
    store, b, _, _ = make_run(tmp_path, items, seed=2, name="b")
    res = compare_runs(store.load(a), store.load(b))
    assert res["items_compared"] + len(res["items_tied_skipped"]) == 12
    assert 0 <= res["mcnemar_p"] <= 1
    for c in res["changed"]:
        assert c["id"] in ("q1", "q2")
    assert store.resolve("a") == a and store.resolve("latest") == b and store.resolve(b[:-1]) == b
    with pytest.raises(LookupError):
        store.resolve("nope")
    assert [r["run_id"] for r in store.runs()] == [a, b]
    store.close()


def test_estimate(items):
    cfg = RunConfig(k=5, price=PRICES["claude-haiku-5-5"])
    est = estimate(cfg, JudgePrompt.from_text(PROMPT), items)
    assert est.calls == 60 and est.cost_usd > 0
