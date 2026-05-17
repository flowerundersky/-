"""LangGraph workflow assembly for code generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph

from .model import ModelBackend, create_backend
from .prompt import (
    OutputMode,
    PromptSpec,
    StyleMode,
    build_code_prompt_template,
    build_revision_prompt_template,
    build_self_check_prompt_template,
)


@dataclass(slots=True)
class GraphRequest:
    """Normalized inputs for a single generation step."""

    task: str
    language: str
    context: str = ""
    output_mode: OutputMode = "code"
    style: StyleMode = "minimal_change"
    constraints: Sequence[str] = field(default_factory=tuple)
    references: Sequence[str] = field(default_factory=tuple)
    draft_code: str | None = None
    self_check: bool = False
    review_result: str = ""
    user_instruction: str = ""
    generation_params: Mapping[str, Any] | None = None


@dataclass(slots=True)
class ReviewOutcome:
    """Structured output from a review step."""

    review_result: str
    next_step: str
    recommend_continue: bool
    notes: str = ""
    raw_output: str = ""


@dataclass(slots=True)
class SelfCheckSession:
    """Finite-state self-check session data."""

    request: GraphRequest
    draft_code: str
    review: ReviewOutcome
    round_index: int = 0
    max_rounds: int = 2


class GraphState(TypedDict, total=False):
    task: str
    language: str
    context: str
    output_mode: OutputMode
    style: StyleMode
    constraints: tuple[str, ...]
    references: tuple[str, ...]
    draft_code: str | None
    self_check: bool
    review_result: str
    user_instruction: str
    generation_params: Mapping[str, Any] | None
    phase: str
    round_index: int
    max_rounds: int
    prompt: Any
    result: str
    review: ReviewOutcome
    output: str


def build_graph(backend: ModelBackend | None = None):
    """Build a LangGraph workflow for the code-generation flow."""

    active_backend = backend or create_backend().load()

    workflow = StateGraph(GraphState)
    workflow.add_node("build_prompt", _build_prompt_node)
    workflow.add_node("generate", _make_generate_node(active_backend))
    workflow.add_node("process", _process_generation_node)
    workflow.add_node("finalize", _finalize_node)
    workflow.add_conditional_edges(
        START,
        _route_prompt,
        {
            "code": "build_prompt",
            "self_check": "build_prompt",
        },
    )
    workflow.add_edge("build_prompt", "generate")
    workflow.add_edge("generate", "process")
    workflow.add_conditional_edges(
        "process",
        _route_after_process,
        {
            "build_prompt": "build_prompt",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("finalize", END)
    return workflow.compile()


def run_graph(
    task: str,
    language: str,
    context: str = "",
    *,
    output_mode: OutputMode = "code",
    style: StyleMode = "minimal_change",
    constraints: Sequence[str] = (),
    references: Sequence[str] = (),
    draft_code: str | None = None,
    self_check: bool = False,
    review_result: str = "",
    user_instruction: str = "",
    backend: ModelBackend | None = None,
    generation_params: Mapping[str, Any] | None = None,
) -> str:
    """Run the LangGraph workflow and return the generated output."""

    return run_graph_with_state(
        task,
        language,
        context,
        output_mode=output_mode,
        style=style,
        constraints=constraints,
        references=references,
        draft_code=draft_code,
        self_check=self_check,
        review_result=review_result,
        user_instruction=user_instruction,
        backend=backend,
        generation_params=generation_params,
    )["output"]


def run_graph_with_state(
    task: str,
    language: str,
    context: str = "",
    *,
    output_mode: OutputMode = "code",
    style: StyleMode = "minimal_change",
    constraints: Sequence[str] = (),
    references: Sequence[str] = (),
    draft_code: str | None = None,
    self_check: bool = False,
    review_result: str = "",
    user_instruction: str = "",
    backend: ModelBackend | None = None,
    generation_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the workflow and return the final graph state."""

    request = GraphRequest(
        task=task,
        language=language,
        context=context,
        output_mode=output_mode,
        style=style,
        constraints=constraints,
        references=references,
        draft_code=draft_code,
        self_check=self_check,
        review_result=review_result,
        user_instruction=user_instruction,
        generation_params=generation_params,
    )

    return build_graph(backend).invoke(_request_to_state(request))


def start_self_check_session(
    request: GraphRequest,
    backend: ModelBackend | None = None,
) -> SelfCheckSession:
    """Start a bounded self-check session and return the first review state."""

    if not request.self_check:
        raise ValueError("self_check must be enabled to start a self-check session")

    state = run_graph_with_state(
        request.task,
        request.language,
        request.context,
        output_mode=request.output_mode,
        style=request.style,
        constraints=request.constraints,
        references=request.references,
        draft_code=request.draft_code,
        self_check=True,
        review_result=request.review_result,
        user_instruction=request.user_instruction,
        backend=backend,
        generation_params=request.generation_params,
    )
    review = state.get("review")
    if not isinstance(review, ReviewOutcome):
        raise ValueError("self-check session did not produce a review outcome")
    return SelfCheckSession(
        request=request,
        draft_code=str(state.get("draft_code", "")),
        review=review,
    )


def continue_self_check_session(
    session: SelfCheckSession,
    backend: ModelBackend | None = None,
    user_instruction: str = "",
) -> SelfCheckSession:
    """Advance a self-check session by one explicit revision round."""

    if session.round_index >= session.max_rounds:
        return session

    state = run_graph_with_state(
        session.request.task,
        session.request.language,
        session.request.context,
        output_mode=session.request.output_mode,
        style=session.request.style,
        constraints=session.request.constraints,
        references=session.request.references,
        draft_code=session.draft_code,
        self_check=True,
        review_result=session.review.review_result,
        user_instruction=user_instruction,
        backend=backend,
        generation_params=session.request.generation_params,
    )
    review = state.get("review")
    if not isinstance(review, ReviewOutcome):
        raise ValueError("self-check session did not produce a review outcome")
    return SelfCheckSession(
        request=session.request,
        draft_code=str(state.get("draft_code", "")),
        review=review,
        round_index=session.round_index + 1,
        max_rounds=session.max_rounds,
    )


def format_review_message(session: SelfCheckSession) -> str:
    """Format the current review state for display to the user."""

    recommendation = "建议继续" if session.review.recommend_continue else "可以结束"
    sections = [
        "自检结果:",
        session.review.review_result.strip() or "无",
        "",
        "下一步建议:",
        session.review.next_step.strip() or "无",
        "",
        "是否建议继续:",
        recommendation,
    ]
    if session.review.notes.strip():
        sections.extend(["", "补充说明:", session.review.notes.strip()])
    sections.extend(["", "当前草稿:", session.draft_code.strip() or "无"])
    return "\n".join(sections)


def _request_to_state(request: GraphRequest) -> GraphState:
    phase = "revision" if request.self_check and request.draft_code and request.review_result.strip() else "generate"
    return GraphState(
        task=request.task,
        language=request.language,
        context=request.context,
        output_mode=request.output_mode,
        style=request.style,
        constraints=tuple(request.constraints),
        references=tuple(request.references),
        draft_code=request.draft_code,
        self_check=request.self_check,
        review_result=request.review_result,
        user_instruction=request.user_instruction,
        generation_params=request.generation_params,
        phase=phase,
    )


def _route_prompt(state: GraphState) -> str:
    return "self_check" if state.get("self_check") else "code"


def _build_prompt_node(state: GraphState) -> dict[str, Any]:
    spec = _state_to_prompt_spec(state)
    phase = state.get("phase", "code")

    if phase == "revision":
        draft_code = state.get("draft_code")
        if not draft_code:
            raise ValueError("draft_code is required when revision phase is enabled")
        review_result = state.get("review_result", "")
        if not str(review_result).strip():
            raise ValueError("review_result is required when revision phase is enabled")

        prompt = build_revision_prompt_template(
            spec,
            draft_code,
            review_result=str(review_result),
            user_instruction=state.get("user_instruction", ""),
        ).invoke({})
        return {"prompt": prompt}

    if phase == "review":
        draft_code = state.get("draft_code")
        if not draft_code:
            raise ValueError("draft_code is required when review phase is enabled")

        prompt = build_self_check_prompt_template(spec, draft_code).invoke({})
        return {"prompt": prompt}

    prompt = build_code_prompt_template(spec).invoke({})
    return {"prompt": prompt}


def _make_generate_node(backend: ModelBackend):
    def _generate(state: GraphState) -> dict[str, Any]:
        result = backend.generate(state["prompt"], params=state.get("generation_params"))
        return {"result": result}

    return _generate


def _process_generation_node(state: GraphState) -> dict[str, Any]:
    phase = state.get("phase", "code")
    result = (state.get("result") or "").strip()

    if phase in {"generate", "revision"} and state.get("self_check"):
        return {
            "draft_code": result,
            "phase": "review",
        }

    if phase in {"generate", "revision"}:
        return {
            "output": result,
            "phase": "done",
        }

    if phase == "review":
        review = _parse_review_outcome(result)
        return {
            "review": review,
            "review_result": review.review_result,
            "output": _format_self_check_output(
                draft_code=state.get("draft_code", ""),
                review=review,
            ),
            "phase": "done",
        }

    return {
        "output": result,
        "phase": "done",
    }


def _route_after_process(state: GraphState) -> str:
    phase = state.get("phase", "done")
    return "build_prompt" if phase == "review" else "finalize"


def _finalize_node(state: GraphState) -> dict[str, Any]:
    output = state.get("output")
    if output:
        return {"output": str(output).strip()}
    return {"output": state.get("result", "").strip()}


def _state_to_prompt_spec(state: GraphState) -> PromptSpec:
    return PromptSpec(
        task=state["task"],
        language=state["language"],
        context=state.get("context", ""),
        output_mode=state.get("output_mode", "code"),
        style=state.get("style", "minimal_change"),
        constraints=state.get("constraints", ()),
        references=state.get("references", ()),
    )


def _without_self_check(request: GraphRequest) -> GraphRequest:
    return GraphRequest(
        task=request.task,
        language=request.language,
        context=request.context,
        output_mode=request.output_mode,
        style=request.style,
        constraints=request.constraints,
        references=request.references,
        draft_code=request.draft_code,
        self_check=False,
        user_instruction=request.user_instruction,
        generation_params=request.generation_params,
    )


def _generate_code(request: GraphRequest, backend: ModelBackend) -> str:
    spec = _request_to_prompt_spec(request)
    prompt = build_code_prompt_template(spec).invoke({})
    return backend.generate(prompt, params=request.generation_params).strip()


def _generate_revision(
    request: GraphRequest,
    draft_code: str,
    review: ReviewOutcome,
    backend: ModelBackend,
    user_instruction: str = "",
) -> str:
    spec = _request_to_prompt_spec(request)
    prompt = build_revision_prompt_template(
        spec,
        draft_code,
        review_result=_render_review_for_revision(review),
        user_instruction=user_instruction,
    ).invoke({})
    return backend.generate(prompt, params=request.generation_params).strip()


def _run_review_step(request: GraphRequest, draft_code: str, backend: ModelBackend) -> ReviewOutcome:
    spec = _request_to_prompt_spec(request)
    prompt = build_self_check_prompt_template(spec, draft_code).invoke({})
    raw_output = backend.generate(prompt, params=request.generation_params).strip()
    return _parse_review_outcome(raw_output)


def _request_to_prompt_spec(request: GraphRequest) -> PromptSpec:
    return PromptSpec(
        task=request.task,
        language=request.language,
        context=request.context,
        output_mode=request.output_mode,
        style=request.style,
        constraints=request.constraints,
        references=request.references,
    )


def _parse_review_outcome(raw_output: str) -> ReviewOutcome:
    parsed: dict[str, str] = {}
    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        key, value = _split_first_colon(line)
        if not key:
            continue
        normalized = key.strip().lower().replace("：", ":")
        parsed[normalized] = value.strip()

    recommend_text = parsed.get("recommend_continue", "")
    recommend_continue = recommend_text.lower() in {"true", "yes", "y", "1", "on", "是"}
    return ReviewOutcome(
        review_result=parsed.get("review_result", raw_output).strip(),
        next_step=parsed.get("next_step", "").strip(),
        recommend_continue=recommend_continue,
        notes=parsed.get("notes", "").strip(),
        raw_output=raw_output,
    )


def _render_review_for_revision(review: ReviewOutcome) -> str:
    lines = [
        f"review_result: {review.review_result.strip() or 'None'}",
        f"next_step: {review.next_step.strip() or 'None'}",
        f"recommend_continue: {'true' if review.recommend_continue else 'false'}",
    ]
    if review.notes.strip():
        lines.append(f"notes: {review.notes.strip()}")
    return "\n".join(lines)


def _format_self_check_output(
    *,
    draft_code: str,
    review: ReviewOutcome,
) -> str:
    recommendation = "建议继续" if review.recommend_continue else "可以结束"
    sections = [
        "自检结果:",
        review.review_result.strip() or "无",
        "",
        "下一步建议:",
        review.next_step.strip() or "无",
        "",
        "是否建议继续:",
        recommendation,
    ]
    if review.notes.strip():
        sections.extend(["", "补充说明:", review.notes.strip()])
    sections.extend(["", "当前草稿:", draft_code.strip() or "无"])
    return "\n".join(sections)


def _split_first_colon(line: str) -> tuple[str, str]:
    if ":" not in line and "：" not in line:
        return "", ""
    if ":" in line:
        return line.split(":", 1)[0], line.split(":", 1)[1]
    return line.split("：", 1)[0], line.split("：", 1)[1]