"""How much of each request a prefix cache could reuse, and what that costs at list prices.

Prefix caches (Anthropic, OpenAI, llama.cpp/Ollama) reuse the longest run of tokens that is
identical to an earlier request, read from the start. This module renders a request the way a
chat template would and measures that shared prefix against the previous request.
"""

from __future__ import annotations

from dataclasses import dataclass

from cache_memory.strategies import Message

CHARS_PER_TOKEN = 4  # rough English average; used for both sides of every comparison
MIN_CACHEABLE_TOKENS = 1024  # Anthropic's minimum cacheable prefix for Sonnet-class models


def render(messages: list[Message]) -> str:
    """A ChatML-style rendering: what the model actually reads, byte for byte."""
    return "".join(f"<|{m.role}|>\n{m.content}\n" for m in messages)


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def tokens(chars: int) -> int:
    return -(-chars // CHARS_PER_TOKEN)


@dataclass(frozen=True)
class Prices:
    """USD per million tokens. Defaults: Claude Sonnet list prices with 5-minute cache writes."""

    input: float = 3.00
    cache_write: float = 3.75
    cache_read: float = 0.30
    output: float = 15.00


@dataclass(frozen=True)
class RequestCost:
    prompt_tokens: int
    cached_tokens: int
    usd: float


def cost_of(prev: str | None, cur: str, out_tokens: int, p: Prices = Prices()) -> RequestCost:
    """Cost of ``cur`` given ``prev`` was the last request: shared prefix read, the rest written."""
    total = tokens(len(cur))
    shared = tokens(common_prefix(prev, cur)) if prev is not None else 0
    if shared < MIN_CACHEABLE_TOKENS:
        shared = 0
    fresh = total - shared
    # Every request writes its full prompt so the next one can read it (a breakpoint on the last
    # message); a provider without explicit writes would charge fresh tokens at the input price.
    usd = (shared * p.cache_read + fresh * p.cache_write + out_tokens * p.output) / 1_000_000
    return RequestCost(total, shared, usd)
