"""Minimal local Ollama client: a reply plus the prefill time that shows whether the cache was hit."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from cache_memory.strategies import Message

DISTILL_PROMPT = """Merge this transcript into the existing session memory.

EXISTING MEMORY:
{existing}

TRANSCRIPT:
{transcript}

RULES:
1. Keep names, IDs, numbers, dates, decisions and open tasks exactly as written.
2. Drop chit-chat and anything finished and no longer relevant.
3. Never guess; mark unknowns "not stated".
4. Output only the updated memory as short Markdown bullets, under 200 words."""


@dataclass(frozen=True)
class Reply:
    text: str
    prompt_tokens: int
    prefill_ms: float
    output_tokens: int


class Ollama:
    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://127.0.0.1:11434", ctx: int = 16384):
        if not host.startswith(("http://127.0.0.1", "http://localhost")):
            raise ValueError("local Ollama only: transcripts never leave this machine")
        self.model, self.host, self.ctx = model, host, ctx

    def chat(self, messages: list[Message], max_tokens: int = 160) -> Reply:
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"num_ctx": self.ctx, "num_predict": max_tokens, "temperature": 0, "seed": 7},
        }
        req = urllib.request.Request(  # noqa: S310 - localhost only, checked in __init__
            f"{self.host}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=600) as r:  # noqa: S310
            d = json.load(r)
        return Reply(
            d["message"]["content"],
            d.get("prompt_eval_count", 0),
            d.get("prompt_eval_duration", 0) / 1e6,
            d.get("eval_count", 0),
        )

    def distill(self, existing: str, turns: list[Message]) -> str:
        transcript = "\n".join(f"{m.role.upper()}: {m.content}" for m in turns)
        prompt = DISTILL_PROMPT.format(existing=existing or "None yet.", transcript=transcript)
        msgs = [Message("system", "You compress chat transcripts into facts."), Message("user", prompt)]
        text = self.chat(msgs, max_tokens=300).text.strip()
        if not text:
            raise RuntimeError("empty distillation")
        return text
