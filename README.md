# cache-friendly-memory

How should a long-running agent chat remember what was said, now that providers cache prompts?

I built the memory my agency's chat agents use, "elephant and goldfish": recent turns stay word for
word (the goldfish), older turns are distilled into a short note in batches (the elephant), and the
note rides on the newest message so everything before it stays byte-identical for the prompt cache.
Then I measured it against the two obvious alternatives, and it did not win where I expected.

| strategy | what it sends |
|---|---|
| `FullHistory` | every turn, forever |
| `RollingSummary` | the usual fix: summarise everything older than the last few turns, every turn, into the system prompt |
| `ElephantGoldfish` | recent turns verbatim; older ones distilled in batches after the reply; the note on the current message only |

![Results table for 40 turns with two distillers, then a cost sweep from 10 to 640 turns](docs/demo.gif)

## Results

A scripted 40-turn agency ops chat (a new client, a GA4 double-counting fix) with eight facts
planted in the first eight turns: a GA4 property ID, a contact, a fee, a deadline and so on. Chat
replies come from Qwen 2.5 7B on my laptop. After turn 40 the model is asked about each fact.
Cost is what the same requests would cost on Claude Sonnet with prompt caching (shared prefix read
at 0.1×, the rest written at 1.25×), plus distillation calls priced at the distiller's own rate.

| strategy | distiller | cost | prompt tokens cached | facts recalled |
|---|---|---|---|---|
| **full history** | none | **$0.146** | 95% | **8/8** |
| rolling summary | Claude Haiku | $0.297 | 64% | 7/8 |
| elephant & goldfish, batch 4 | Claude Haiku | $0.225 | 69% | 7/8 |
| elephant & goldfish, batch 12 | Claude Haiku | $0.195 | 79% | 8/8 |
| rolling summary | Qwen 7B, local (free) | $0.168 | 69% | **0/8** |
| elephant & goldfish, batch 4 | Qwen 7B, local (free) | $0.173 | 71% | 1/8 |
| elephant & goldfish, batch 12 | Qwen 7B, local (free) | $0.168 | 80% | 2/8 |

What this says:

1. **With prompt caching, sending everything was cheapest at 40 turns and the only option that
   never forgot.** Its prompt grows to 7,400 tokens, but 95% of what it sends was sent before and is
   read from cache at a tenth of the price. Measured on the laptop too: median prefill 62 ms
   against 460–870 ms for the strategies that keep rewriting their prompt.
2. **Compression pays later than I assumed.** `bench/sweep.py` replays 10 to 640 turns through the
   same cost model with no model calls. Full history is cheapest up to 80 turns; by 160 turns
   elephant & goldfish (batch 12) is cheaper, and at 640 turns it costs $3.42 against $12.26.
3. **The distiller decides whether memory works at all.** With a 7B local model doing the
   distilling (batch 12), the first note held the 5 facts it had seen; the next merge dropped the
   property ID, the contact and the fee, and the final note held 1 of 8. Along the way it gave the
   client another client's reporting day, and the final note says no duplicate purchases were found
   when the chat said 30% were double counted. With Claude Haiku distilling, the same strategies
   kept 7 or 8 facts. Every note from both runs is in `bench/results_*.json`.
4. **Batch size is the cache lever.** The hub's setting (batch 4) distils every second turn, and
   every distillation breaks the cached prefix. Batch 12 cut distillations from 18 to 7 and was
   cheaper and more accurate with either distiller.

## What I changed because of it

- Below roughly 80 turns, don't compress: send the history and let the cache do the work.
- When compressing, distil in large batches with a model that keeps facts, not the cheapest local
  model, and keep the note on the current message so the prefix stays stable.
- Check the note, not just the replies: in the failing runs the replies stayed fluent while the
  facts underneath were gone.

## Use it

```python
from cache_memory import ElephantGoldfish, render

memory = ElephantGoldfish(distill=my_distiller, window=8, batch=12)
messages = memory.build(system_prompt, user_message)  # send these
memory.record(user_message, reply)  # after the reply; may distil
```

`distill(existing_note, turns) -> new_note` is any callable; if it raises or returns nothing, the
turns stay verbatim and nothing is lost. `cache_memory.cost_of(prev, cur, output_tokens)` prices a
request against the one before it.

## Reproduce

```bash
uv run pytest                                            # 16 tests, 100% coverage
uv run python bench/run.py                               # needs Ollama with qwen2.5:7b, ~10 min
uv run python bench/run.py --distiller claude-haiku --only rolling_summary elephant_goldfish_b4 elephant_goldfish_b12
uv run python bench/sweep.py                             # cost only, seconds
uv run python bench/report.py
```

## Limits

- One scripted conversation, one quiz, temperature 0. Recall is from eight facts, so 7/8 and 8/8
  are close.
- Costs are modelled from Anthropic's published caching rules and list prices, applied to the real
  requests; they were not billed. Token counts use 4 characters per token.
- The sweep uses fixed message lengths (the measured averages) and does not model recall.
- Prefill times are from one laptop GPU and Ollama's single cache slot.

## Where it came from

The elephant & goldfish memory runs in my agency's operator hub, distilling with a local Qwen model
in batches of 4. This repo is a clean Python version with the benchmark the hub never had. The
benchmark says to change both settings. MIT licence.
