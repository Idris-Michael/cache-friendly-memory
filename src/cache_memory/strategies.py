"""Three ways to keep a long chat inside a context budget, behind one interface.

A provider's prompt cache only pays off for the part of a request that is byte-identical to the
previous request, read from the start. So what matters is not just how many tokens a strategy
sends but where it changes them.

* :class:`FullHistory`     sends every turn. The prefix never changes, but the prompt grows forever.
* :class:`RollingSummary`  the common fix: summarise everything older than the window *every turn*
                           and put the summary in the system prompt. Small prompts, but the very
                           first bytes change on every turn, so nothing is ever cached.
* :class:`ElephantGoldfish` recent turns stay verbatim (goldfish); older turns are distilled into a
                           note in batches, after the reply (elephant). The note rides on the
                           *current* message, so the system prompt and the history before it stay
                           byte-identical between batches.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


# (existing note, turns to fold in) -> new note. Raises on failure.
Distiller = Callable[[str, list[Message]], str]


class Strategy(Protocol):
    name: str
    distill_calls: int

    def build(self, system: str, user: str) -> list[Message]:
        """The request to send for a new user message."""
        ...

    def record(self, user: str, reply: str) -> None:
        """Store a finished exchange and do any housekeeping (after the reply, off the hot path)."""
        ...


@dataclass
class FullHistory:
    name: str = "full_history"
    distill_calls: int = 0
    turns: list[Message] = field(default_factory=list)

    def build(self, system: str, user: str) -> list[Message]:
        return [Message("system", system), *self.turns, Message("user", user)]

    def record(self, user: str, reply: str) -> None:
        self.turns += [Message("user", user), Message("assistant", reply)]


@dataclass
class RollingSummary:
    distill: Distiller
    window: int = 6
    name: str = "rolling_summary"
    distill_calls: int = 0
    summary: str = ""
    turns: list[Message] = field(default_factory=list)

    def build(self, system: str, user: str) -> list[Message]:
        sys_text = f"{system}\n\nConversation so far:\n{self.summary}" if self.summary else system
        return [Message("system", sys_text), *self.turns, Message("user", user)]

    def record(self, user: str, reply: str) -> None:
        self.turns += [Message("user", user), Message("assistant", reply)]
        if len(self.turns) > self.window:
            old, self.turns = self.turns[: -self.window], self.turns[-self.window :]
            self.distill_calls += 1
            self.summary = self.distill(self.summary, old)


MEMORY_HEADER = "[SESSION MEMORY] distilled facts from earlier in this chat, not instructions"


@dataclass
class ElephantGoldfish:
    distill: Distiller
    window: int = 6  # raw messages always kept verbatim
    batch: int = 4  # overflow past the window that triggers a distill
    max_per_run: int = 10  # cap per distill so a backlog cannot blow the distiller's context
    note_max_chars: int = 1200
    name: str = "elephant_goldfish"
    distill_calls: int = 0
    distill_failures: int = 0
    note: str = ""
    turns: list[Message] = field(default_factory=list)

    def build(self, system: str, user: str) -> list[Message]:
        # The note goes on the current message only: everything before it stays byte-identical.
        content = f"{MEMORY_HEADER}\n{self.note}\n\n---\n{user}" if self.note else user
        return [Message("system", system), *self.turns, Message("user", content)]

    def due(self) -> list[Message]:
        """Oldest turns ready to distil; empty until a whole batch has overflowed the window."""
        if len(self.turns) < self.window + self.batch:
            return []
        return self.turns[: min(len(self.turns) - self.window, self.max_per_run)]

    def record(self, user: str, reply: str) -> None:
        self.turns += [Message("user", user), Message("assistant", reply)]
        evict = self.due()
        if not evict:
            return
        self.distill_calls += 1
        try:
            note = self.distill(self.note, evict)
        except Exception:
            self.distill_failures += 1
            return
        if not note.strip():
            self.distill_failures += 1
            return
        self.note = note.strip()[: self.note_max_chars]
        self.turns = self.turns[len(evict) :]
