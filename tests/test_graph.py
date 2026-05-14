"""Tests for the LangGraph workflow."""

from src.graph import (
    GraphRequest,
    build_graph,
    continue_self_check_session,
    run_graph,
    start_self_check_session,
)


class DummyBackend:
    def __init__(self) -> None:
        self.prompts: list[object] = []
        self.params: list[dict | None] = []

    def load(self) -> "DummyBackend":
        return self

    def generate(self, prompt, params=None) -> str:
        self.prompts.append(prompt)
        self.params.append(params)
        return "generated text"


class SequenceBackend:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[object] = []

    def load(self) -> "SequenceBackend":
        return self

    def generate(self, prompt, params=None) -> str:
        self.prompts.append(prompt)
        _ = params
        return self.responses.pop(0)


def test_run_graph_builds_prompt_and_calls_backend() -> None:
    backend = DummyBackend()

    result = run_graph(
        task="Add a health check endpoint.",
        language="Python",
        context="Use FastAPI.",
        references=("src/app.py",),
        backend=backend,
        generation_params={"max_new_tokens": 16},
    )

    assert result == "generated text"
    assert backend.params == [{"max_new_tokens": 16}]
    prompt_text = backend.prompts[0].to_string()
    assert "You are a precise code generation assistant." in prompt_text
    assert "Add a health check endpoint." in prompt_text
    assert "src/app.py" in prompt_text


def test_run_graph_self_check_uses_review_prompt() -> None:
    backend = DummyBackend()

    result = run_graph(
        task="Review this adapter.",
        language="Python",
        self_check=True,
        draft_code="return {'ok': True}",
        backend=backend,
    )

    assert result == "generated text"
    prompt_text = backend.prompts[0].to_string()
    assert "You are a precise code review assistant." in prompt_text
    assert "return {'ok': True}" in prompt_text


def test_build_graph_returns_compiled_workflow() -> None:
    workflow = build_graph(DummyBackend())

    assert hasattr(workflow, "invoke")


def test_self_check_session_advances_only_on_explicit_continue() -> None:
    backend = SequenceBackend(
        [
            "draft code",
            "review_result: needs cleanup\nnext_step: fix naming\nrecommend_continue: true\nnotes: keep going",
            "revised code",
            "review_result: looks good\nnext_step: stop\nrecommend_continue: false\nnotes: done",
        ]
    )
    request = GraphRequest(
        task="Review this adapter.",
        language="Python",
        self_check=True,
    )

    session = start_self_check_session(request, backend=backend)

    assert session.draft_code == "draft code"
    assert session.review.review_result == "needs cleanup"
    assert session.review.recommend_continue is True

    next_session = continue_self_check_session(session, backend=backend)

    assert next_session.round_index == 1
    assert next_session.draft_code == "revised code"
    assert next_session.review.review_result == "looks good"
    assert next_session.review.recommend_continue is False
