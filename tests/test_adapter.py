"""Tests for the parser-to-graph request adapter."""

from src.adapter import GraphRequestPayload, build_graph_request


def test_build_graph_request_normalizes_structured_payload() -> None:
    payload = {
        "task": "  Implement login  ",
        "language": " Python ",
        "context": "  Use FastAPI.  ",
        "output_mode": "code",
        "style": "minimal_change",
        "constraints": "Keep the response minimal.\nPrefer async.",
        "references": [" src/app.py ", "docs/README.md"],
        "generation_params": {"max_new_tokens": 64},
    }

    request = build_graph_request(payload)

    assert request.task == "Implement login"
    assert request.language == "Python"
    assert request.context == "Use FastAPI."
    assert request.output_mode == "code"
    assert request.style == "minimal_change"
    assert request.constraints == ("Keep the response minimal.", "Prefer async.")
    assert request.references == ("src/app.py", "docs/README.md")
    assert request.generation_params == {"max_new_tokens": 64}
    assert request.self_check is False
    assert request.draft_code is None


def test_build_graph_request_allows_self_check_without_draft_code() -> None:
    payload = {
        "task": "Review this code.",
        "language": "Python",
        "self_check": True,
    }

    request = build_graph_request(payload)

    assert request.self_check is True
    assert request.draft_code is None


def test_graph_request_payload_can_be_converted_directly() -> None:
    payload = GraphRequestPayload(
        task="  Summarize this module  ",
        language="  Python  ",
        context="  Use concise prose.  ",
        constraints=(" Keep it short ",),
        references=(" src/chain.py ",),
    )

    request = build_graph_request(payload)

    assert request.task == "Summarize this module"
    assert request.language == "Python"
    assert request.context == "Use concise prose."
    assert request.constraints == ("Keep it short",)
    assert request.references == ("src/chain.py",)
