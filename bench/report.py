"""Print the 40-turn results from the saved runs as one table.

full_history comes from results_full.json (no distiller involved); the compressing strategies come
from results_haiku_distill.json and results_qwen_distill.json.

    uv run python bench/report.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
RUNS = [
    ("results_full.json", "-"),
    ("results_haiku_distill.json", "Claude Haiku"),
    ("results_qwen_distill.json", "Qwen 2.5 7B (local)"),
]


def main() -> None:
    print(
        "| strategy | distiller | cost (Sonnet prices) | cached | last prompt | median prefill | distills | recall |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for file, distiller in RUNS:
        path = HERE / file
        if not path.exists():
            continue
        for r in json.loads(path.read_text(encoding="utf-8"))["results"]:
            print(
                f"| {r['strategy']} | {distiller} | ${r['total_usd']:.3f} | {r['cache_hit_share']:.0%} | "
                f"{r['last_prompt_tokens']:,} tok | {r['median_prefill_ms']:.0f} ms | {r['distill_calls']} | "
                f"{r['recall']}/{r['recall_of']} |"
            )


if __name__ == "__main__":
    main()
