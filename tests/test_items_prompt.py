import json

import pytest

from judgejitter.items import Item, ItemsError, load_items
from judgejitter.prompt import JudgePrompt, PromptError, output_schema, parse_structured, parse_text


def test_load_jsonl_with_aliases(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text(
        '{"id": "a", "input": "q1", "output": "o1", "reference": "r1"}\n'
        "\n"
        '{"question": "q2", "answer": "o2"}\n'
        '{"id": 7, "prompt": "q3", "response": {"text": "o3"}, "expected": "r3"}\n'
    )
    items = load_items(p)
    assert [i.id for i in items] == ["a", "3", "7"]
    assert items[1] == Item("3", "q2", "o2", None)
    assert items[2].output == '{"text": "o3"}' and items[2].reference == "r3"


def test_load_jsonl_errors(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "a", "input": "q"}\n')
    with pytest.raises(ItemsError, match="no output"):
        load_items(p)
    p.write_text('{"id": "a", "output": "x"}\n{"id": "a", "output": "y"}\n')
    with pytest.raises(ItemsError, match="duplicate id"):
        load_items(p)
    p.write_text("not json\n")
    with pytest.raises(ItemsError, match="line 1"):
        load_items(p)
    (tmp_path / "x.txt").write_text("")
    with pytest.raises(ItemsError, match="extension"):
        load_items(tmp_path / "x.txt")


def test_load_csv(tmp_path):
    p = tmp_path / "items.csv"
    p.write_text('id,input,output,reference\n1,What is 2+2?,4,4\n2,"Capital, France",Paris,\n', encoding="utf-8")
    items = load_items(p)
    assert items[1] == Item("2", "Capital, France", "Paris", None)


def test_load_promptfoo_v3_shape(tmp_path):
    data = {
        "evalId": "eval-1",
        "results": {
            "version": 3,
            "results": [
                {
                    "testIdx": 0,
                    "promptIdx": 0,
                    "prompt": {"raw": "Translate: hello", "label": "p"},
                    "vars": {"text": "hello", "expected": "bonjour"},
                    "response": {"output": "bonjour"},
                    "provider": {"id": "anthropic:messages:claude-haiku-5-5"},
                },
                {
                    "testIdx": 1,
                    "promptIdx": 0,
                    "prompt": {"raw": "Translate: cat"},
                    "vars": {"text": "cat"},
                    "testCase": {"assert": [{"type": "llm-rubric", "value": "Is French for cat"}]},
                    "response": {"output": "chat"},
                    "provider": {"id": "anthropic:messages:claude-haiku-5-5"},
                },
                {"testIdx": 2, "promptIdx": 0, "prompt": {"raw": "x"}, "response": None, "error": "timeout"},
            ],
        },
    }
    p = tmp_path / "results.json"
    p.write_text(json.dumps(data))
    items = load_items(p)
    assert [i.id for i in items] == ["t0-p0", "t1-p0"]
    assert items[0] == Item("t0-p0", "Translate: hello", "bonjour", "bonjour")
    assert items[1].reference == "Is French for cat"


def test_load_promptfoo_outputs_shape_and_providers(tmp_path):
    row = {"promptIdx": 0, "prompt": {"raw": "q"}, "response": {"output": "a"}}
    data = {
        "results": {
            "outputs": [
                {**row, "testIdx": 0, "provider": {"id": "m1"}},
                {**row, "testIdx": 0, "provider": {"id": "m2"}},
            ]
        }
    }
    p = tmp_path / "out.json"
    p.write_text(json.dumps(data))
    assert [i.id for i in load_items(p)] == ["t0-p0-m1", "t0-p0-m2"]


def test_prompt_render_and_errors():
    p = JudgePrompt.from_text("Q: {{ input }}\nA: {{ output }}\nRef: {{ reference }}")
    assert p.render(Item("1", "q", "a", None)) == "Q: q\nA: a\nRef: "
    assert len(p.sha) == 12
    with pytest.raises(PromptError, match="placeholders"):
        JudgePrompt.from_text("no placeholders here")
    bad = JudgePrompt.from_text("{{ ouput }}")
    with pytest.raises(PromptError, match="ouput"):
        bad.render(Item("1", "q", "a"))


def test_output_schema():
    s = output_schema("pass-fail")
    assert s["properties"]["verdict"]["enum"] == ["PASS", "FAIL"]
    assert s["required"] == ["reason", "verdict"] and s["additionalProperties"] is False
    assert output_schema("1-5")["properties"]["score"]["enum"] == [1, 2, 3, 4, 5]


def test_parse_structured():
    assert parse_structured('{"reason": "r", "verdict": "pass"}', "pass-fail").verdict == "PASS"
    assert parse_structured('{"reason": "r", "verdict": "MAYBE"}', "pass-fail").verdict is None
    assert parse_structured("nope", "pass-fail").error == "output was not valid JSON"
    assert parse_structured('{"reason": "r", "score": 4}', "1-5").verdict == 4
    assert parse_structured('{"reason": "r", "score": "5"}', "1-5").verdict == 5
    assert parse_structured('{"reason": "r", "score": 9}', "1-5").verdict is None
    assert parse_structured('{"reason": "r", "score": true}', "1-5").verdict is None


def test_parse_text():
    assert parse_text("The answer is wrong.\nVerdict: FAIL", "pass-fail").verdict == "FAIL"
    assert parse_text("It could fail, but overall: pass", "pass-fail").verdict == "PASS"
    assert parse_text("no idea", "pass-fail").verdict is None
    assert parse_text("Reasoning about 2 issues.\nScore: 4/5", "1-5").verdict == 4
    assert parse_text("Clear and complete.\n5", "1-5").verdict == 5
    assert parse_text("3.5 out of 10 things", "1-5").verdict is None
