"""Replay the 40-turn script through four memory strategies on local Qwen and measure:

* modelled cost per chat at Claude Sonnet list prices, from the byte-identical prefix each
  request shares with the one before (how prefix caching works at Anthropic and OpenAI)
* measured prefill time on Ollama, the local evidence of the same effect
* distillation calls, and how many of the eight planted facts the model recalls after turn 40

    uv run python bench/run.py            # all four strategies, ~20 minutes on a laptop GPU
    uv run python bench/run.py --turns 6  # smoke test
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from script import QUIZ, SYSTEM, TURNS  # noqa: E402

from cache_memory import ElephantGoldfish, FullHistory, Prices, RollingSummary, cost_of, render  # noqa: E402
from cache_memory.cache import tokens  # noqa: E402
from cache_memory.ollama import Ollama  # noqa: E402


def make(name: str, llm: Ollama):  # type: ignore[no-untyped-def]
    return {
        "full_history": lambda: FullHistory(),
        "rolling_summary": lambda: RollingSummary(llm.distill, window=6),
        "elephant_goldfish_b4": lambda: ElephantGoldfish(llm.distill, window=6, batch=4),
        "elephant_goldfish_b12": lambda: ElephantGoldfish(llm.distill, window=8, batch=12),
    }[name]()


HAIKU = Prices(input=1.00, cache_write=1.25, cache_read=0.10, output=5.00)


class ClaudeDistiller:
    """Distils with Claude Haiku through the claude CLI (subscription; tools off). Same prompt as Ollama."""

    def __init__(self, model: str = "claude-haiku-5-5") -> None:
        self.model = model

    def distill(self, existing, turns):  # type: ignore[no-untyped-def]
        import shutil
        import subprocess

        from cache_memory.ollama import DISTILL_PROMPT

        transcript = "\n".join(f"{m.role.upper()}: {m.content}" for m in turns)
        prompt = DISTILL_PROMPT.format(existing=existing or "None yet.", transcript=transcript)
        cli = shutil.which("claude") or "claude"
        args = [
            cli,
            "-p",
            "--model",
            self.model,
            "--output-format",
            "text",
            "--tools",
            "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--setting-sources",
            "",
            "--no-session-persistence",
            "--system-prompt",
            "You compress chat transcripts into facts.",
        ]
        out = subprocess.run(
            args,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=240,
            check=True,
        ).stdout.strip()
        if not out:
            raise RuntimeError("empty distillation")
        return out


class Metered:
    """Wraps the distiller to cost each call (uncached) at the distiller's own prices and keep every note."""

    def __init__(self, inner, prices: Prices) -> None:  # type: ignore[no-untyped-def]
        self.inner, self.prices, self.usd, self.notes = inner, prices, 0.0, []

    def distill(self, existing, turns):  # type: ignore[no-untyped-def]
        note = self.inner.distill(existing, turns)
        prompt_chars = len(existing) + sum(len(m.content) for m in turns) + 600  # + instructions
        self.usd += cost_of(None, "x" * prompt_chars, tokens(len(note)), self.prices).usd
        self.notes.append(note)
        return note


def run_one(name: str, llm: Ollama, turns: list[str], distiller: str) -> dict[str, object]:
    # Local distillation costs nothing; a hosted cheap model is priced at Haiku list rates.
    meter = Metered(llm, Prices(0, 0, 0, 0)) if distiller == "ollama" else Metered(ClaudeDistiller(), HAIKU)
    s = make(name, meter)  # type: ignore[arg-type]
    prev: str | None = None
    usd = 0.0
    prompt_tok = cached_tok = 0
    prefill: list[float] = []
    per_turn = []
    started = time.perf_counter()
    for i, user in enumerate(turns):
        msgs = s.build(SYSTEM, user)
        cur = render(msgs)
        r = llm.chat(msgs, max_tokens=160)
        c = cost_of(prev, cur, r.output_tokens)
        usd += c.usd
        prompt_tok += c.prompt_tokens
        cached_tok += c.cached_tokens
        prefill.append(r.prefill_ms)
        per_turn.append(
            {
                "turn": i + 1,
                "prompt_tokens": c.prompt_tokens,
                "cached_tokens": c.cached_tokens,
                "prefill_ms": round(r.prefill_ms, 1),
            }
        )
        prev = cur
        s.record(user, r.text)
        print(
            f"  {name} turn {i + 1}: {c.prompt_tokens} tok, {c.cached_tokens} cached, "
            f"{r.prefill_ms:.0f} ms prefill",
            flush=True,
        )

    distill_usd = meter.usd
    recalled = []
    for q, accepted in QUIZ:
        ans = llm.chat(s.build(SYSTEM, q), max_tokens=60).text
        recalled.append({"q": q, "answer": ans, "ok": any(a.lower() in ans.lower() for a in accepted)})
    return {
        "strategy": name,
        "distiller": distiller if s.distill_calls else None,
        "notes": meter.notes,
        "turns": len(turns),
        "chat_usd": round(usd, 5),
        "distill_usd": round(distill_usd, 5),
        "total_usd": round(usd + distill_usd, 5),
        "prompt_tokens": prompt_tok,
        "cached_tokens": cached_tok,
        "cache_hit_share": round(cached_tok / prompt_tok, 3) if prompt_tok else 0,
        "last_prompt_tokens": per_turn[-1]["prompt_tokens"],
        "median_prefill_ms": round(statistics.median(prefill), 1),
        "distill_calls": s.distill_calls,
        "recall": sum(r["ok"] for r in recalled),
        "recall_of": len(QUIZ),
        "wall_s": round(time.perf_counter() - started, 1),
        "per_turn": per_turn,
        "quiz": recalled,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=len(TURNS))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--distiller", choices=["ollama", "claude-haiku"], default="ollama")
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    llm = Ollama(a.model)
    names = a.only or ["full_history", "rolling_summary", "elephant_goldfish_b4", "elephant_goldfish_b12"]
    print(f"system prompt: {tokens(len(SYSTEM))} tokens")
    results = [run_one(n, llm, TURNS[: a.turns], a.distiller) for n in names]
    out = {
        "model": a.model,
        "distiller": a.distiller,
        "system_tokens": tokens(len(SYSTEM)),
        "results": results,
    }
    (HERE / a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print("| strategy | modelled cost | cached share | last prompt | median prefill | distills | recall |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['strategy']} | ${r['total_usd']:.4f} | {r['cache_hit_share']:.0%} | "
            f"{r['last_prompt_tokens']} tok | {r['median_prefill_ms']} ms | {r['distill_calls']} | "
            f"{r['recall']}/{r['recall_of']} |"
        )


if __name__ == "__main__":
    main()
