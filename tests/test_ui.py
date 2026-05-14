"""Tests for UI composition."""

from src.ui import _format_field_block, build_ui


def test_ui_builds_without_launching() -> None:
    assert build_ui() is not None


def test_field_block_formatter_renders_lists_and_scalars() -> None:
    fields = {
        "task": "改 parser",
        "constraints": ["尽量少改", "保留结构"],
        "self_check": True,
    }

    rendered = _format_field_block(fields)

    assert "task: 改 parser" in rendered
    assert "constraints:" in rendered
    assert "- 尽量少改" in rendered
    assert "self_check: True" in rendered