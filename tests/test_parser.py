"""Tests for the natural-language parser."""

from src.parser import ParsedRequestPayload, build_structured_fields, build_structured_fields_with_trace, parse_natural_language_request


class DummyBackend:
	def __init__(self, raw_output: str) -> None:
		self.raw_output = raw_output

	def generate(self, prompt, params=None) -> str:
		_ = prompt
		_ = params
		return self.raw_output


def test_parse_natural_language_request_extracts_core_fields() -> None:
	text = "你给我写一个 pipeline 风格的 adapter，用 Python 语言写，尽量少改，输出 patch，参考 src/graph.py"

	payload = parse_natural_language_request(
		text,
		semantic_backend=DummyBackend(
"""task: write an adapter
language: python
context: adapter for pipeline style code generation
output_mode: patch
style: minimal_change
constraints:
references:
- src/graph.py
draft_code:
self_check: false
""",
		),
	)

	assert isinstance(payload, ParsedRequestPayload)
	assert payload.task == "write an adapter"
	assert payload.language == "python"
	assert payload.style == "minimal_change"
	assert payload.output_mode == "patch"
	assert payload.references == ("src/graph.py",)


def test_build_structured_fields_parses_semantic_output() -> None:
	text = "帮我写一个代码生成器"
	payload = build_structured_fields(
		text,
		semantic_backend=DummyBackend(
"""task: write a code generator
language: python
context:
output_mode: patch
style: minimal_change
constraints:
references:
draft_code:
self_check: false
""",
		),
	)

	assert payload.task == "write a code generator"
	assert payload.language == "python"
	assert payload.output_mode == "patch"


def test_semantic_field_parser_uses_field_lines_only() -> None:
	from src.parser import _parse_semantic_fields

	raw_output = """task: write an adapter
language: typescript
constraints:
- keep structure
- avoid unrelated refactors
references:
- src/parser.py
self_check: true
"""
	parsed = _parse_semantic_fields(raw_output)

	assert parsed["language"] == "typescript"
	assert parsed["constraints"] == ["keep structure", "avoid unrelated refactors"]
	assert parsed["references"] == ["src/parser.py"]
	assert parsed["self_check"] is True


def test_semantic_field_parser_normalizes_decorated_keys() -> None:
	from src.parser import _parse_semantic_fields

	raw_output = """**task**: write an adapter
**language**: python
**output_mode**: patch
**style**: strict_format
**self_check**: false
"""
	parsed = _parse_semantic_fields(raw_output)

	assert parsed["task"] == "write an adapter"
	assert parsed["language"] == "python"
	assert parsed["output_mode"] == "patch"
	assert parsed["style"] == "strict_format"
	assert parsed["self_check"] is False


def test_semantic_field_parser_normalizes_decorated_values() -> None:
	from src.parser import _parse_semantic_fields

	raw_output = """task: write an adapter
language: python
output_mode: `code`
style: **strict_format**
self_check: `false`
"""
	parsed = _parse_semantic_fields(raw_output)

	assert parsed["output_mode"] == "code"
	assert parsed["style"] == "strict_format"
	assert parsed["self_check"] is False


def test_semantic_field_parser_recognizes_review_aliases() -> None:
	from src.parser import _parse_semantic_fields

	raw_output = """task: write an adapter
language: python
output_mode: patch
style: minimal_change
review: yes
"""
	parsed = _parse_semantic_fields(raw_output)

	assert parsed["self_check"] is True


def test_structured_trace_exposes_semantic_raw_output() -> None:
	trace = build_structured_fields_with_trace(
		"请帮我改成 typescript",
		semantic_backend=DummyBackend(
			"""task: write an adapter
language: typescript
context:
output_mode: code
style: minimal_change
constraints:
references:
draft_code:
self_check: false
""",
		),
	)

	assert "task: write an adapter" in trace.semantic_raw_output
	assert trace.final_payload.language == "typescript"


def test_semantic_extract_rejects_chatty_output() -> None:
	from pytest import raises

	with raises(ValueError, match="unexpected text before task field") as exc_info:
		parse_natural_language_request(
			"今天很开心，我想写一python程序",
			semantic_backend=DummyBackend("Hello! I'm Qwen, created by Alibaba Cloud."),
		)

	assert "Hello! I'm Qwen, created by Alibaba Cloud." in str(exc_info.value)