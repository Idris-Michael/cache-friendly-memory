import itertools
import json

import pytest

from cache_memory import ElephantGoldfish, FullHistory, Message, Prices, RollingSummary, cost_of, render
from cache_memory.cache import MIN_CACHEABLE_TOKENS, common_prefix, tokens
from cache_memory.ollama import Ollama
from cache_memory.strategies import MEMORY_HEADER

SYSTEM = "You are the ops assistant. " * 300  # long enough to clear the cacheable minimum


def fake_distill(existing: str, turns: list[Message]) -> str:
    return (existing + " | " if existing else "") + ",".join(m.content for m in turns)


def drive(strategy, n: int) -> list[str]:
    """Run n turns and return each rendered request."""
    out = []
    for i in range(n):
        out.append(render(strategy.build(SYSTEM, f"u{i}")))
        strategy.record(f"u{i}", f"a{i}")
    return out


def prefix_shared(reqs: list[str]) -> list[int]:
    return [common_prefix(a, b) for a, b in itertools.pairwise(reqs)]


def rewriting_distill():
    """Like a real model: every distillation rewrites the note from the top."""
    n = {"v": 0}

    def d(existing: str, turns: list[Message]) -> str:
        n["v"] += 1
        return f"note version {n['v']}: " + ",".join(m.content for m in turns)

    return d


def breaks(reqs: list[str]) -> int:
    """Requests whose history differs from the previous request's (anything before the new message)."""
    return sum(not cur.startswith(prev[: prev.rfind("<|user|>")]) for prev, cur in itertools.pairwise(reqs))


@pytest.mark.parametrize(("batch", "expected_distills"), [(4, 13), (12, 5)])
def test_elephant_goldfish_only_breaks_the_prefix_when_it_distils(batch, expected_distills):
    eg = ElephantGoldfish(rewriting_distill(), batch=batch)
    reqs = drive(eg, 30)
    assert eg.distill_calls == expected_distills
    # Every break is a distillation, and nothing else breaks the prefix.
    assert breaks(reqs) == eg.distill_calls


def test_a_bigger_batch_trades_a_longer_window_for_fewer_cache_breaks():
    small = ElephantGoldfish(rewriting_distill(), batch=4)
    big = ElephantGoldfish(rewriting_distill(), batch=12)
    assert breaks(drive(small, 30)) > 2 * breaks(drive(big, 30))


def test_rolling_summary_changes_the_system_prompt_on_every_turn_once_it_starts():
    rs = RollingSummary(rewriting_distill())
    reqs = drive(rs, 20)
    assert rs.distill_calls == 17
    # From the first summary on, every request diverges inside the system message.
    assert all(s < len("<|system|>\n") + len(SYSTEM) + 40 for s in prefix_shared(reqs)[4:])
    assert breaks(reqs) == 16


def test_full_history_never_breaks_the_prefix_but_grows_without_limit():
    fh = FullHistory()
    reqs = drive(fh, 20)
    assert all(b.startswith(a[: a.rfind("<|user|>")]) for a, b in itertools.pairwise(reqs))
    assert len(reqs[-1]) > len(reqs[0]) + 200


def test_memory_rides_on_the_current_message_only():
    eg = ElephantGoldfish(fake_distill)
    drive(eg, 6)
    msgs = eg.build(SYSTEM, "now")
    assert msgs[0].content == SYSTEM
    assert msgs[-1].content.startswith(MEMORY_HEADER) and msgs[-1].content.endswith("now")
    assert all(MEMORY_HEADER not in m.content for m in msgs[:-1])


def test_the_window_is_never_distilled_and_nothing_is_lost():
    eg = ElephantGoldfish(fake_distill)
    drive(eg, 5)  # 10 messages = window 6 + batch 4: first distillation
    assert eg.note.startswith("u0,a0,u1,a1")
    assert [m.content for m in eg.turns] == ["u2", "a2", "u3", "a3", "u4", "a4"]


def test_a_failed_distillation_keeps_the_turns_verbatim():
    def boom(existing: str, turns: list[Message]) -> str:
        raise TimeoutError("ollama down")

    eg = ElephantGoldfish(boom)
    drive(eg, 8)
    assert eg.distill_failures == eg.distill_calls > 0
    assert len(eg.turns) == 16 and eg.note == ""


def test_an_empty_distillation_is_treated_as_a_failure():
    eg = ElephantGoldfish(lambda e, t: "   ")
    drive(eg, 5)
    assert eg.distill_failures == 1 and len(eg.turns) == 10


def test_a_backlog_is_distilled_in_capped_slices():
    eg = ElephantGoldfish(fake_distill, max_per_run=4)
    eg.turns = [Message("user", str(i)) for i in range(20)]
    assert len(eg.due()) == 4


def test_the_note_is_capped():
    eg = ElephantGoldfish(lambda e, t: "x" * 5000, note_max_chars=100)
    drive(eg, 5)
    assert len(eg.note) == 100


def test_cost_reads_the_shared_prefix_and_writes_the_rest():
    a = "s" * (MIN_CACHEABLE_TOKENS * 4 * 2)
    b = a + "t" * 400
    first = cost_of(None, a, 0)
    second = cost_of(a, b, 0)
    assert first.cached_tokens == 0
    assert second.cached_tokens == tokens(len(a))
    p = Prices()
    assert second.usd == pytest.approx((tokens(len(a)) * p.cache_read + 100 * p.cache_write) / 1e6)


def test_a_prefix_below_the_provider_minimum_is_not_cached():
    assert cost_of("abc", "abcd", 10).cached_tokens == 0


def test_ollama_client_refuses_remote_hosts():
    with pytest.raises(ValueError, match="local Ollama only"):
        Ollama(host="https://api.example.com")


def test_ollama_client_parses_a_reply_and_distils(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {
                    "message": {"content": "- client: Halden"},
                    "prompt_eval_count": 120,
                    "prompt_eval_duration": 5_000_000,
                    "eval_count": 9,
                }
            ).encode()

    def fake_urlopen(req, timeout):
        seen["body"] = json.loads(req.data)
        return Resp()

    monkeypatch.setattr("cache_memory.ollama.urllib.request.urlopen", fake_urlopen)
    o = Ollama()
    r = o.chat([Message("user", "hi")])
    assert (r.text, r.prompt_tokens, r.prefill_ms, r.output_tokens) == ("- client: Halden", 120, 5.0, 9)
    assert seen["body"]["options"]["temperature"] == 0
    assert o.distill("", [Message("user", "client is Halden")]) == "- client: Halden"


def test_an_empty_ollama_distillation_raises(monkeypatch):
    o = Ollama()
    monkeypatch.setattr(o, "chat", lambda m, max_tokens=0: type("R", (), {"text": " "})())
    with pytest.raises(RuntimeError, match="empty"):
        o.distill("", [Message("user", "x")])
