"""Judge prompt rendering, the output schema, and verdict parsing."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from jinja2 import Environment, StrictUndefined, TemplateError

from .items import Item

SCALES = ("pass-fail", "1-5")


class PromptError(ValueError):
    pass


@dataclass
class JudgePrompt:
    template: str
    sha: str

    @classmethod
    def from_text(cls, text: str) -> JudgePrompt:
        if "{{" not in text:
            raise PromptError("the judge prompt has no placeholders; use {{ input }}, {{ output }} and {{ reference }}")
        try:
            _env().from_string(text)
        except TemplateError as e:
            raise PromptError(f"the judge prompt is not a valid template: {e}") from None
        return cls(text, hashlib.sha256(text.encode()).hexdigest()[:12])

    def render(self, item: Item) -> str:
        try:
            return (
                _env()
                .from_string(self.template)
                .render(input=item.input, output=item.output, reference=item.reference or "", id=item.id)
            )
        except TemplateError as e:
            raise PromptError(f"item {item.id}: {e}") from None


def _env() -> Environment:
    # StrictUndefined: a typo like {{ ouput }} fails loudly instead of rendering empty.
    return Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)


def output_schema(scale: str) -> dict:
    """JSON schema for structured output: a short reason, then the verdict."""
    if scale == "pass-fail":
        verdict = {"type": "string", "enum": ["PASS", "FAIL"]}
        name = "verdict"
    elif scale == "1-5":
        verdict = {"type": "integer", "enum": [1, 2, 3, 4, 5]}
        name = "score"
    else:
        raise PromptError(f"unknown scale {scale!r}")
    return {
        "type": "object",
        "properties": {"reason": {"type": "string"}, name: verdict},
        "required": ["reason", name],
        "additionalProperties": False,
    }


@dataclass
class Parsed:
    verdict: str | int | None
    reason: str | None
    error: str | None = None


_LABEL_RE = re.compile(r"\b(PASS|FAIL)\b", re.IGNORECASE)
_SCORE_LINE_RE = re.compile(r"\b(?:score|rating|grade)\b\D{0,12}([1-5])\b", re.IGNORECASE)
_DIGIT_RE = re.compile(r"(?<![\d.])([1-5])(?![\d.])")


def parse_structured(text: str, scale: str) -> Parsed:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return Parsed(None, None, "output was not valid JSON")
    if not isinstance(data, dict):
        return Parsed(None, None, "output was not a JSON object")
    reason = data.get("reason")
    if scale == "pass-fail":
        v = data.get("verdict")
        if isinstance(v, str) and v.strip().upper() in ("PASS", "FAIL"):
            return Parsed(v.strip().upper(), reason)
        return Parsed(None, reason, f"verdict {v!r} is not PASS or FAIL")
    v = data.get("score")
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        return Parsed(None, reason, f"score {v!r} is not 1 to 5")
    try:
        n = int(str(v).strip())
    except ValueError:
        return Parsed(None, reason, f"score {v!r} is not 1 to 5")
    if 1 <= n <= 5:
        return Parsed(n, reason)
    return Parsed(None, reason, f"score {v!r} is not 1 to 5")


def parse_text(text: str, scale: str) -> Parsed:
    """Parse a free-text judge reply. The last verdict in the text wins."""
    if not text:
        return Parsed(None, None, "empty output")
    if scale == "pass-fail":
        found = _LABEL_RE.findall(text)
        if not found:
            return Parsed(None, text, "no PASS or FAIL in the output")
        return Parsed(found[-1].upper(), text)
    found = _SCORE_LINE_RE.findall(text)
    if not found:
        last_line = text.strip().splitlines()[-1]
        found = _DIGIT_RE.findall(last_line)
    if not found:
        return Parsed(None, text, "no score from 1 to 5 in the output")
    return Parsed(int(found[-1]), text)
