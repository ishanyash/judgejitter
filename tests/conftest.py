import json
import random
from types import SimpleNamespace

import pytest

from judgejitter.items import Item

PROMPT = """Question: {{ input }}
Reference: {{ reference }}
Answer: {{ output }}
Reply PASS if the answer agrees with the reference, else FAIL."""


@pytest.fixture
def items():
    return [Item(id=f"q{i}", input=f"question {i}", output=f"answer {i}", reference=f"ref {i}") for i in range(12)]


def fake_response(text: str, *, stop="end_turn", input_tokens=500, output_tokens=60, thinking=True):
    content = []
    if thinking:
        content.append(SimpleNamespace(type="thinking", thinking="", signature="sig"))
    content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=0,
        ),
        stop_reason=stop,
        model="claude-haiku-5-5",
    )


class FakeJudge:
    """Passes even-numbered items, fails odd ones, and flips q1 and q2 sometimes."""

    def __init__(self, seed=0, scale="pass-fail", fail_first=None):
        self.rng = random.Random(seed)
        self.calls = []
        self.scale = scale
        self.fail_first = fail_first

    async def __call__(self, body):
        self.calls.append(body)
        if self.fail_first is not None:
            raise self.fail_first
        content = body["messages"][0]["content"]
        n = int(content.split("question ")[1].split()[0])
        if self.scale == "1-5":
            score = 5 if n % 2 == 0 else 2
            if n in (1, 2) and self.rng.random() < 0.5:
                score = 3
            return fake_response(json.dumps({"reason": "ok", "score": score}))
        verdict = "PASS" if n % 2 == 0 else "FAIL"
        if n in (1, 2) and self.rng.random() < 0.4:
            verdict = "FAIL" if verdict == "PASS" else "PASS"
        return fake_response(json.dumps({"reason": f"because {n}", "verdict": verdict}))


class StatusError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
