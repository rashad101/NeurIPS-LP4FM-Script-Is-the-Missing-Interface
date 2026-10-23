"""Prompt construction — the part that must **not** vary across conditions.

The task wrapper (how to format an answer, how options are labelled) is held
constant **in English**. Only the item content changes between conditions. If
the answer-format instruction were itself romanised, a measured drop could be
the model failing to parse *our* instruction rather than the item, and the
"isolate orthography" claim would not hold.

The safety task gets **no wrapper at all**: any scaffolding changes refusal
behaviour, and H4 is about the model's own behaviour on the bare prompt.
"""

from __future__ import annotations

from typing import Any

_LETTERS = "ABCDEFGH"

Message = dict[str, str]

MCQ_INSTRUCTION = "Answer with the letter of the correct option only."
NUMBER_INSTRUCTION = (
    "Solve the problem and give the final numeric answer on the last line "
    "in the form `Answer: <number>`."
)


def format_options(options: list[str] | None) -> str:
    if not options:
        return ""
    return "\n".join(f"{_LETTERS[i]}. {option}" for i, option in enumerate(options))


def build_prompt(item: Any) -> str:
    """The user-turn text for one rendered cell."""
    if item.answer_type == "refusal":
        # No wrapper: scaffolding changes refusal behaviour.
        return item.prompt

    if item.answer_type == "mcq":
        return "\n\n".join(
            part for part in (item.prompt, format_options(item.options), MCQ_INSTRUCTION) if part
        )

    if item.answer_type == "number":
        return f"{item.prompt}\n\n{NUMBER_INSTRUCTION}"

    if item.answer_type == "constraints":
        # The constraints live in the item's meta rather than in its prompt, so
        # the transforms never rewrite them — the model is asked for the same
        # structure in every condition.
        instructions = list((item.meta or {}).get("instructions") or [])
        if instructions:
            body = "\n".join(f"- {line}" for line in instructions)
            return f"{item.prompt}\n\nFollow these requirements exactly:\n{body}"
        return item.prompt

    return item.prompt


def build_messages(item: Any) -> list[Message]:
    """Chat-format messages for one rendered cell.

    A single user turn, with no system prompt: system-prompt support varies
    across the roster, and an instruction that reaches some models and not
    others would be a per-model confound.
    """
    return [{"role": "user", "content": build_prompt(item)}]
