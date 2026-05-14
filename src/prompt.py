"""LangChain-native prompt templates and assembly utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import dedent
from typing import Literal, Sequence

from langchain_core.prompts import ChatPromptTemplate


OutputMode = Literal["code", "markdown", "patch"]
StyleMode = Literal["minimal_change", "preserve_structure", "strict_format"]


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """Structured inputs for prompt assembly."""

    task: str
    language: str
    context: str = ""
    output_mode: OutputMode = "code"
    style: StyleMode = "minimal_change"
    constraints: Sequence[str] = field(default_factory=tuple)
    references: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ReviewSpec:
    """Structured inputs for a review or revision step."""

    task: str
    language: str
    context: str = ""
    output_mode: OutputMode = "code"
    style: StyleMode = "minimal_change"
    constraints: Sequence[str] = field(default_factory=tuple)
    references: Sequence[str] = field(default_factory=tuple)


LANGUAGE_RULES: dict[str, tuple[str, ...]] = {
    "python": (
        "Use snake_case for variables and functions.",
        "Prefer pathlib, dataclasses, and explicit type hints where they improve clarity.",
        "Keep imports explicit and grouped by standard library, third-party, and local modules.",
    ),
    "javascript": (
        "Use modern ES syntax and prefer const over let unless reassignment is required.",
        "Keep functions small and avoid unnecessary abstraction.",
        "Match the project's existing formatting conventions if context is provided.",
    ),
    "typescript": (
        "Use explicit types at boundaries and avoid implicit any.",
        "Prefer narrow interfaces and discriminated unions for structured data.",
        "Keep runtime behavior and type declarations aligned.",
    ),
    "shell": (
        "Prefer set -euo pipefail for safety when appropriate.",
        "Quote variables and avoid word-splitting bugs.",
        "Keep commands POSIX-friendly unless Bash-specific features are required.",
    ),
}


OUTPUT_MODE_RULES: dict[OutputMode, tuple[str, ...]] = {
    "code": (
        "Return only code unless the caller explicitly asks for explanation.",
        "Do not wrap the response in markdown fences.",
    ),
    "markdown": (
        "Wrap code in a fenced markdown block with the correct language tag.",
        "Keep prose minimal and place it outside the code block only if necessary.",
    ),
    "patch": (
        "Return a unified diff or apply_patch-style patch.",
        "Keep file paths explicit and only include changed hunks.",
    ),
}


STYLE_RULES: dict[StyleMode, tuple[str, ...]] = {
    "minimal_change": (
        "Make the smallest safe change that satisfies the request.",
        "Avoid unrelated refactors and preserve the existing code shape.",
    ),
    "preserve_structure": (
        "Preserve the current module and function structure where possible.",
        "Reuse existing names, types, and control flow unless a change is required.",
    ),
    "strict_format": (
        "Follow the requested output format exactly.",
        "Do not add extra commentary unless the request allows it.",
    ),
}


def build_language_rules(language: str) -> tuple[str, ...]:
    """Return the language-specific rule bundle for a target language."""

    return LANGUAGE_RULES.get(
        language.strip().lower(),
        (
            "Match the project's existing style and naming conventions.",
            "Keep the solution focused and avoid unnecessary abstractions.",
        ),
    )


def build_code_prompt_template(spec: PromptSpec) -> ChatPromptTemplate:
    """Build the LangChain chat prompt template for code generation."""

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a precise code generation assistant.
Follow the user's requirements exactly.
Prefer the smallest correct change.
If information is missing, state the minimum assumption needed.
Treat the user's natural-language description as the source of truth for output format, language, and style.
Do not ask the user to choose among preset options.
If the request does not explicitly require self-check, return the final result directly.

Language Rules:
{language_rules}

Output Rules:
{output_rules}

Style Rules:
{style_rules}""",
            ),
            (
                "human",
                """Task:
{task}

Language:
{language}

Context:
{context}

Constraints:
{constraints}

References:
{references}""",
            ),
        ]
    ).partial(
        language_rules=_format_rules(build_language_rules(spec.language)),
        output_rules=_format_rules(OUTPUT_MODE_RULES.get(spec.output_mode, OUTPUT_MODE_RULES["code"])),
        style_rules=_format_rules(STYLE_RULES.get(spec.style, STYLE_RULES["minimal_change"])),
        task=spec.task,
        language=spec.language,
        context=_normalize_text(spec.context) or "None",
        constraints=_format_list(spec.constraints) or "None",
        references=_format_list(spec.references) or "None",
    )


def build_self_check_prompt_template(spec: PromptSpec, draft_code: str) -> ChatPromptTemplate:
    """Build the LangChain chat prompt template for self-check review."""

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a precise code review assistant.
Verify requirement coverage.
Check syntax and obvious runtime safety issues.
Explain what passed, what failed, and what should happen next.
Do not decide whether to continue on behalf of the user.
The result must support a bounded review loop where the user explicitly chooses whether to continue.

Language Rules:
{language_rules}

Output Rules:
{output_rules}

Style Rules:
{style_rules}""",
            ),
            (
                "human",
                """Task:
{task}

Language:
{language}

Context:
{context}

Draft:
{draft_code}

Constraints:
{constraints}

Return a structured review with these fields only:
review_result: <what passed or failed>
next_step: <what should happen next>
recommend_continue: <true|false>
notes: <short extra guidance or blank>""",
            ),
        ]
    ).partial(
        language_rules=_format_rules(build_language_rules(spec.language)),
        output_rules=_format_rules(OUTPUT_MODE_RULES.get(spec.output_mode, OUTPUT_MODE_RULES["code"])),
        style_rules=_format_rules(STYLE_RULES.get(spec.style, STYLE_RULES["minimal_change"])),
        task=spec.task,
        language=spec.language,
        context=_normalize_text(spec.context) or "None",
        draft_code=_normalize_text(draft_code) or "None",
        constraints=_format_list(spec.constraints) or "None",
    )


def build_revision_prompt_template(
    spec: PromptSpec,
    draft_code: str,
    review_result: str,
    user_instruction: str = "",
) -> ChatPromptTemplate:
    """Build the prompt template for an explicit revision round."""

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a precise code generation assistant.
Revise the supplied draft based on the review result and the user's latest instruction.
Return only the updated code.
This is one bounded revision step inside a user-controlled review loop, not an autonomous loop.
Do not continue iterating unless the user explicitly asks for another round.

Language Rules:
{language_rules}

Output Rules:
{output_rules}

Style Rules:
{style_rules}""",
            ),
            (
                "human",
                """Task:
{task}

Language:
{language}

Context:
{context}

Draft:
{draft_code}

Review Result:
{review_result}

User Instruction:
{user_instruction}

Constraints:
{constraints}

References:
{references}""",
            ),
        ]
    ).partial(
        language_rules=_format_rules(build_language_rules(spec.language)),
        output_rules=_format_rules(OUTPUT_MODE_RULES.get(spec.output_mode, OUTPUT_MODE_RULES["code"])),
        style_rules=_format_rules(STYLE_RULES.get(spec.style, STYLE_RULES["minimal_change"])),
        task=spec.task,
        language=spec.language,
        context=_normalize_text(spec.context) or "None",
        draft_code=_normalize_text(draft_code) or "None",
        review_result=_normalize_text(review_result) or "None",
        user_instruction=_normalize_text(user_instruction) or "None",
        constraints=_format_list(spec.constraints) or "None",
        references=_format_list(spec.references) or "None",
    )


def _format_rules(rules: Sequence[str]) -> str:
    return "\n".join(f"- {rule.strip()}" for rule in rules if rule and rule.strip())


def _format_list(items: Sequence[str]) -> str:
    return "\n".join(f"- {item.strip()}" for item in items if item and item.strip())


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in dedent(text).strip().splitlines() if line.strip())
