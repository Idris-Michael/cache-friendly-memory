"""Where does compressing history start to pay? A cost-only sweep, no model calls.

Replays conversations of 10 to 640 turns through each strategy with fixed-length user messages and
replies (the averages measured in the 40-turn run) and a stand-in distiller whose note is the
measured average length. Chat requests are priced at Claude Sonnet list rates with prefix caching;
distillations at Claude Haiku rates, uncached. Recall is not modelled here: see results*.json.

    uv run python bench/sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from script import SYSTEM  # noqa: E402

from cache_memory import ElephantGoldfish, FullHistory, Prices, RollingSummary, cost_of, render  # noqa: E402
from cache_memory.cache import tokens  # noqa: E402
from cache_memory.strategies import Message  # noqa: E402

HAIKU = Prices(input=1.00, cache_write=1.25, cache_read=0.10, output=5.00)
USER_CHARS, REPLY_CHARS, NOTE_CHARS = 70, 560, 900  # measured averages from the 40-turn run
CONTEXT_LIMIT = 200_000


def make(name: str, spent: list[float]):  # type: ignore[no-untyped-def]
    def distill(existing: str, turns: list[Message]) -> str:
        chars = len(existing) + sum(len(m.content) for m in turns) + 600
        spent[0] += cost_of(None, "x" * chars, tokens(NOTE_CHARS), HAIKU).usd
        return f"v{spent[0]:.9f} " + "n" * NOTE_CHARS  # a fresh note each time, like a real distiller

    return {
        "full_history": lambda: FullHistory(),
        "rolling_summary": lambda: RollingSummary(distill, window=6),
        "elephant_goldfish_b4": lambda: ElephantGoldfish(distill, window=6, batch=4),
        "elephant_goldfish_b12": lambda: ElephantGoldfish(distill, window=8, batch=12),
    }[name]()


def run(name: str, n: int) -> dict[str, float | int | bool]:
    spent = [0.0]
    s = make(name, spent)
    prev, usd, peak = None, 0.0, 0
    for i in range(n):
        user = f"turn {i} " + "u" * USER_CHARS
        cur = render(s.build(SYSTEM, user))
        c = cost_of(prev, cur, tokens(REPLY_CHARS))
        usd += c.usd
        peak = max(peak, c.prompt_tokens)
        prev = cur
        s.record(user, f"reply {i} " + "r" * REPLY_CHARS)
    return {
        "usd": round(usd + spent[0], 4),
        "peak_prompt_tokens": peak,
        "fits_context": peak <= CONTEXT_LIMIT,
    }


def main() -> None:
    names = ["full_history", "rolling_summary", "elephant_goldfish_b4", "elephant_goldfish_b12"]
    turns = [10, 20, 40, 80, 160, 320, 640]
    table = {n: {name: run(name, n) for name in names} for n in turns}
    (HERE / "sweep_results.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print("| turns | " + " | ".join(names) + " | cheapest |")
    print("|---" * (len(names) + 2) + "|")
    for n, row in table.items():
        cells = [f"${r['usd']:.2f}" + ("" if r["fits_context"] else " (over 200k)") for r in row.values()]
        best = min(row, key=lambda k: float(row[k]["usd"]))
        print(f"| {n} | " + " | ".join(cells) + f" | {best} |")


if __name__ == "__main__":
    main()
