"""List prices in US dollars per million tokens.

Checked on 1 November 2026 at https://platform.claude.com/docs/en/about-claude/pricing.
Pass --price-in and --price-out to use other prices or another model.
"""

from __future__ import annotations

from dataclasses import dataclass

CHECKED = "2026-11-01"


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_write: float | None = None  # 5-minute cache write
    cache_read: float | None = None
    # Haiku 5.5 charges more when one request's prompt is over 100k tokens.
    long_threshold: int | None = None
    long_input: float | None = None
    long_output: float | None = None


PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(0.10, 0.50, 0.125, 0.01, 100_000, 0.50, 2.50),
    "claude-sonnet-5-5": Price(2.0, 10.0, 2.50, 0.10),
    "claude-opus-5-5": Price(4.0, 20.0),
    "claude-fable-5-1": Price(10.0, 50.0),
}


def price_for(model: str, price_in: float | None = None, price_out: float | None = None) -> Price | None:
    if price_in is not None and price_out is not None:
        return Price(price_in, price_out)
    return PRICES.get(model)


def call_cost(
    price: Price | None,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float | None:
    if price is None:
        return None
    prompt_tokens = input_tokens + cache_write_tokens + cache_read_tokens
    long = price.long_threshold is not None and prompt_tokens > price.long_threshold
    p_in = price.long_input if long else price.input
    p_out = price.long_output if long else price.output
    p_cw = price.cache_write if price.cache_write is not None else p_in * 1.25
    p_cr = price.cache_read if price.cache_read is not None else p_in * 0.1
    if long:
        p_cw, p_cr = p_cw * 5, p_cr * 5  # same 5x step as input and output
    return (
        input_tokens * p_in + output_tokens * p_out + cache_write_tokens * p_cw + cache_read_tokens * p_cr
    ) / 1_000_000
