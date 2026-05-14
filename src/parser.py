"""Hybrid natural-language parser for extracting structured graph fields.

Layering:
- Rule layer extracts stable, explicit signals first.
- Small semantic LLM layer can fill ambiguous fields later.
- Pydantic validates the final structured fields.

This module only turns free-form UI text into structured fields.
The adapter layer is responsible for converting those fields into
the GraphRequest used by the workflow layer.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Sequence

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .model import GenerationParams, ModelBackend, QwenHFBackend
from .prompt import OutputMode, StyleMode


logger = logging.getLogger(__name__)


# Small semantic extraction model used for weakly structured field filling.
SMALL_SEMANTIC_LLM_MODEL_PATH = (
"/home/user/Qwen_codegen_web/models/Qwen2.5-3B/models--Qwen--Qwen2.5-3B-Instruct-AWQ/snapshots/3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd"

)


SEMANTIC_EXTRACTION_GENERATION_PARAMS: dict[str, Any] = {
	"max_new_tokens": 256,
	"min_new_tokens": 16,
	"temperature": 0.0,
	"top_p": 1.0,
	"repetition_penalty": 1.0,
	"do_sample": False,
}


SEMANTIC_EXTRACTION_FIELDS = (
	"task",
	"language",
	"context",
	"output_mode",
	"style",
	"constraints",
	"references",
	"draft_code",
	"self_check",
)


SEMANTIC_FIELD_ALIASES: dict[str, str] = {
	"task": "task",
	"任务": "task",
	"language": "language",
	"languages": "language",
	"lang": "language",
	"langauge": "language",
	"语言": "language",
	"context": "context",
	"上下文": "context",
	"output_mode": "output_mode",
	"outputmode": "output_mode",
	"output mode": "output_mode",
	"输出模式": "output_mode",
	"format": "output_mode",
	"输出格式": "output_mode",
	"diff": "output_mode",
	"style": "style",
	"风格": "style",
	"stylemode": "style",
	"写法": "style",
	"constraints": "constraints",
	"constraint": "constraints",
	"约束": "constraints",
	"references": "references",
	"reference": "references",
	"参考": "references",
	"draft_code": "draft_code",
	"draft": "draft_code",
	"草稿": "draft_code",
	"self_check": "self_check",
	"selfcheck": "self_check",
	"review": "self_check",
	"check": "self_check",
	"审查": "self_check",
	"检查": "self_check",
	"自检": "self_check",
}


LIST_FIELDS = {"constraints", "references","context"}

SEMANTIC_OUTPUT_FIELDS: tuple[str, ...] = SEMANTIC_EXTRACTION_FIELDS
SEMANTIC_REQUIRED_FIELDS: tuple[str, ...] = ("task", "language")


class SemanticOutputContractError(ValueError):
	def __init__(self, message: str, raw_output: str) -> None:
		super().__init__(message)
		self.raw_output = raw_output

	def __str__(self) -> str:
		base_message = super().__str__()
		raw_output = self.raw_output.strip()
		if not raw_output:
			return base_message
		return f"{base_message}\n原始输出：\n{raw_output}"


class StructuredParseTrace(BaseModel):
	"""Intermediate parser outputs for UI inspection."""

	rule_fields: dict[str, Any]
	semantic_raw_output: str
	semantic_fields: dict[str, Any]
	final_payload: ParsedRequestPayload


class ParsedRequestPayload(BaseModel):
	"""Validated structured fields extracted from natural language."""

	model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

	task: str
	language: str
	context: str = ""
	output_mode: OutputMode = "code"
	style: StyleMode = "minimal_change"
	constraints: Sequence[str] = Field(default_factory=tuple)
	references: Sequence[str] = Field(default_factory=tuple)
	draft_code: str | None = None
	self_check: bool = False
	generation_params: dict[str, Any] = Field(default_factory=dict)

	@field_validator("task", "language")
	@classmethod
	def _normalize_required_text(cls, value: Any) -> str:
		text = str(value).strip()
		if not text:
			raise ValueError("field cannot be empty")
		return text

	@field_validator("context", mode="before")
	@classmethod
	def _normalize_context(cls, value: Any) -> str:
		if value is None:
			return ""
		return str(value).strip()

	@field_validator("draft_code", mode="before")
	@classmethod
	def _normalize_draft_code(cls, value: Any) -> str | None:
		if value is None:
			return None
		text = str(value).strip()
		return text or None

	@field_validator("constraints", "references", mode="before")
	@classmethod
	def _normalize_sequence(cls, value: Any) -> tuple[str, ...]:
		if value is None:
			return ()

		if isinstance(value, str):
			raw_items = re.split(r"[\n,;]+", value)
		elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
			raw_items = value
		else:
			raw_items = (value,)

		items = [str(item).strip() for item in raw_items if str(item).strip()]
		return tuple(items)

	@field_validator("generation_params", mode="before")
	@classmethod
	def _normalize_generation_params(cls, value: Any) -> dict[str, Any]:
		if value is None:
			return {}
		if isinstance(value, Mapping):
			return dict(value)
		raise TypeError("generation_params must be a mapping")

	@model_validator(mode="after")
	def _validate_self_check(self) -> "ParsedRequestPayload":
		if self.self_check and not self.draft_code:
			raise ValueError("draft_code is required when self_check is enabled")
		return self



def build_structured_fields(
	text: str,
	*,
	semantic_backend: ModelBackend | None = None,
) -> ParsedRequestPayload:
	"""Public helper for UI callers that want a structured payload."""

	return parse_natural_language_request(text, semantic_backend=semantic_backend)


def parse_natural_language_request(
	text: str,
	*,
	semantic_backend: ModelBackend | None = None,
) -> ParsedRequestPayload:

	return build_structured_fields_with_trace(text, semantic_backend=semantic_backend).final_payload


def build_structured_fields_with_trace(
	text: str,
	*,
	semantic_backend: ModelBackend | None = None,
) -> StructuredParseTrace:
	rule_fields: dict[str, Any] = {}
	semantic_raw_output, semantic_fields = _semantic_extract(text, backend=semantic_backend)
	final_fields = semantic_fields

	return StructuredParseTrace(
		rule_fields=rule_fields,
		semantic_raw_output=semantic_raw_output,
		semantic_fields=semantic_fields,
		final_payload=ParsedRequestPayload.model_validate(final_fields),
	)

def _semantic_extract(
	raw_text: str,
	*,
	backend: ModelBackend | None = None,
) -> tuple[str, dict[str, Any]]:
	semantic_backend = backend or QwenHFBackend(model_path=SMALL_SEMANTIC_LLM_MODEL_PATH).load()

	prompt_value = _build_semantic_extraction_prompt(raw_text).invoke({})

	generation_params = GenerationParams.from_input(SEMANTIC_EXTRACTION_GENERATION_PARAMS)
	raw_output = semantic_backend.generate(prompt_value, params=generation_params.to_generation_kwargs())
	try:
		_validate_semantic_output_contract(raw_output)
	except ValueError as exc:
		raise SemanticOutputContractError(str(exc), raw_output) from exc

	parsed_fields = _parse_semantic_fields(raw_output)
	logger.debug(
		"semantic extraction end: raw_output_len=%s parsed_keys=%s raw_output_preview=%r",
		len(raw_output or ""),
		sorted(parsed_fields.keys()),
		(raw_output or "")[:200].replace("\n", "\\n"),
	)
	return raw_output, parsed_fields


def _build_semantic_extraction_prompt(raw_text: str) -> ChatPromptTemplate:
	return ChatPromptTemplate.from_messages(
		[
			(
				"system",
				"""你是一个严格的字段抽取器，不是聊天助手。
你的唯一任务是把用户原始文本转换成固定字段块，并且必须完全遵守下面的输出协议。
你要根据自然语言自行归一化输出形式、目标语言、风格和自检意图，不要要求用户先做选项选择。
如果某个字段没有显式写出，就结合上下文推断最合适的默认值，不要先让用户在候选项之间手动选择。

硬性约束：
- 只输出字段块，不要解释，不要寒暄，不要分析，不要总结。
- 不要输出 Markdown 代码块，不要输出引号包裹的文本，不要输出表格。
- 不要对字段名或字段值使用 Markdown 加粗、斜体、行内代码或其他装饰。
- 字段名必须是纯文本，不能写成 `**task**`、`__task__`、``task``、`*task*`这类形式。
- 第一条非空行必须是 `task:`。
- 最后一条非空行必须是 `self_check:`。
- 不要输出任何额外字段，字段名不可变更，不要重命名，不要合并字段。
- 每个普通字段必须占一行，格式必须是 `key: value`。
- `constraints` 和 `references` 只能使用列表格式，先写 `constraints:` 或 `references:`，再写 `- item`。
- 如果某个字段未知，保留该字段，但值留空；列表字段则保留空列表区域，不要编造内容。
- 如果用户原文里没有明确答案，优先结合上下文给出最合理的默认值；只有完全无法判断时才留空。

必须严格按这个顺序输出：
task: <short task summary>
language: <normalized language name>
context: <optional context>
output_mode: <normalized output contract>
style: <normalized style>
constraints:
- <constraint 1>
- <constraint 2>
references:
- <reference 1>
- <reference 2>
draft_code: <optional draft code or blank>
self_check: <true|false>

字段说明：
- task: 用一句话概括用户要做什么。
- language: 直接输出最匹配的语言名，不要解释。
- context: 只放用户明确给出的背景信息。
- output_mode: 输出最匹配的结果形式，例如 code、markdown 或 patch。
- style: 输出最匹配的风格，例如 minimal_change、preserve_structure 或 strict_format。
- constraints: 只放显式约束，不要补充推断。
- references: 只放用户明确提到的文件或路径，没有参考时表格下写"- 无参考"。
- draft_code: 只有用户已经给出草稿时才填。
- self_check: 只有用户明确要求自检、审查、复核、检查时才填 true，否则 false。

校验规则：
- 不要输出不在协议中的任何文本。
- 不要在字段之间插入额外说明。
- 不要使用嵌套列表，除 `constraints` 和 `references` 之外不能出现 `- `。
- 如果无法满足协议，依然只能输出协议字段，不能切换成对话回答。""",
			),
			(
				"human",
				"""原始文本：

{raw_text}""",
			),
		],
	).partial(raw_text=raw_text)


def _validate_semantic_output_contract(raw_output: str) -> None:
	text = raw_output.strip()
	if not text:
		raise ValueError("semantic model returned empty output")

	lines = [line.strip() for line in text.splitlines() if line.strip()]
	if not lines:
		raise ValueError("semantic model returned only whitespace")

	if lines[0].startswith("```") or lines[-1].startswith("```"):
		raise ValueError("semantic model output must not use code fences")

	expected_index = 0
	current_list_key: str | None = None
	seen_field = False

	for line in lines:
		key, value = _split_field_line(line)
		if key is not None:
			seen_field = True
			current_list_key = None
			normalized_key = _normalize_semantic_key(key)
			if normalized_key is None:
				raise ValueError(f"unexpected field name in semantic output: {key}")

			if expected_index >= len(SEMANTIC_OUTPUT_FIELDS):
				raise ValueError(f"unexpected extra field in semantic output: {normalized_key}")

			expected_key = SEMANTIC_OUTPUT_FIELDS[expected_index]
			if normalized_key != expected_key:
				raise ValueError(
					f"semantic output field order mismatch: expected {expected_key}, got {normalized_key}",
				)

			if normalized_key in LIST_FIELDS and value:
				raise ValueError(f"list field {normalized_key} must not contain inline values")

			expected_index += 1
			if normalized_key in LIST_FIELDS:
				current_list_key = normalized_key
			continue

		if current_list_key is None:
			if not seen_field:
				raise ValueError("unexpected text before task field")
			raise ValueError("semantic output contains text outside the declared list fields")

		if not line.startswith(("- ", "* ")):
			raise ValueError(f"invalid list item in semantic output: {line}")

	if expected_index != len(SEMANTIC_OUTPUT_FIELDS):
		missing_fields = SEMANTIC_OUTPUT_FIELDS[expected_index:]
		raise ValueError(f"semantic output is missing fields: {', '.join(missing_fields)}")

	parsed_preview = _parse_field_lines(text)
	missing_required = [field for field in SEMANTIC_REQUIRED_FIELDS if not str(parsed_preview.get(field, "")).strip()]
	if missing_required:
		raise ValueError(f"semantic output is missing required fields: {', '.join(missing_required)}")


def _parse_semantic_fields(raw_output: str) -> dict[str, Any]:
	text = _strip_code_fences(raw_output)
	return _parse_field_lines(text)


def _parse_field_lines(text: str) -> dict[str, Any]:
	parsed: dict[str, Any] = {}
	current_list_key: str | None = None
	current_mapping_key: str | None = None
	current_mapping: dict[str, Any] = {}
	started = False

	for raw_line in text.splitlines():
		line = raw_line.strip()
		if not line:
			continue

		key, value = _split_field_line(line)
		if key is not None:
			started = True
			normalized_key = _normalize_semantic_key(key)
			if normalized_key is None:
				current_list_key = None
				current_mapping_key = None
				continue

			if normalized_key in LIST_FIELDS:
				current_mapping_key = None
				if value:
					parsed[normalized_key] = _coerce_semantic_value(normalized_key, value)
					current_list_key = normalized_key
				else:
					parsed[normalized_key] = []
					current_list_key = normalized_key
				continue

			current_list_key = None
			if normalized_key == "draft_code":
				current_mapping_key = None
				parsed[normalized_key] = value
				continue

			if normalized_key == "self_check":
				current_mapping_key = None
				parsed[normalized_key] = _coerce_semantic_value(normalized_key, value)
				continue

			if normalized_key == "generation_params":
				current_mapping_key = normalized_key
				current_mapping = {}
				if value:
					current_mapping["value"] = value
				parsed[normalized_key] = current_mapping
				continue

			parsed[normalized_key] = value
			current_mapping_key = None
			continue

		if not started:
			continue

		if current_list_key is not None and line.startswith(("- ", "* ")):
			parsed.setdefault(current_list_key, []).append(line[2:].strip())
			continue

		if current_mapping_key == "generation_params" and ":" in line:
			inner_key, inner_value = _split_first_colon(line)
			if inner_key:
				current_mapping[inner_key] = _coerce_scalar(inner_value)
				parsed[current_mapping_key] = current_mapping

	return _normalize_semantic_mapping(parsed)


def _normalize_semantic_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
	normalized: dict[str, Any] = {}
	for key, value in payload.items():
		normalized_key = _normalize_semantic_key(str(key))
		if normalized_key is None:
			continue
		normalized[normalized_key] = _coerce_semantic_value(normalized_key, value)
	return normalized


def _normalize_semantic_key(raw_key: str) -> str | None:
	key = raw_key.strip().lower().replace("：", ":")
	key = re.sub(r"\s+", " ", key)
	return SEMANTIC_FIELD_ALIASES.get(key)


def _coerce_semantic_value(key: str, value: Any) -> Any:
	if value is None:
		return None

	if key in LIST_FIELDS:
		if isinstance(value, str):
			items = [item.strip() for item in re.split(r"[\n,;]+", value) if item.strip()]
			return items
		if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
			return [str(item).strip() for item in value if str(item).strip()]
		return [str(value).strip()] if str(value).strip() else []

	if key == "self_check":
		if isinstance(value, bool):
			return value
		text = _strip_markdown_decorations(str(value)).lower()
		if text in {"true", "yes", "y", "1", "on", "是"}:
			return True
		if text in {"false", "no", "n", "0", "off", "否"}:
			return False
		return bool(text)

	if isinstance(value, Mapping):
		return {str(k).strip(): _coerce_scalar(v) for k, v in value.items()}

	if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
		return [str(item).strip() for item in value if str(item).strip()]

	return _coerce_scalar(value)


def _coerce_scalar(value: Any) -> Any:
	if value is None:
		return None
	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)):
		return value
	return _strip_markdown_decorations(str(value))


def _strip_markdown_decorations(text: str) -> str:
	stripped = text.strip()
	decorated_pairs = (("**", "**"), ("__", "__"), ("`", "`"))
	changed = True
	while changed:
		changed = False
		for prefix, suffix in decorated_pairs:
			if len(stripped) >= len(prefix) + len(suffix) and stripped.startswith(prefix) and stripped.endswith(suffix):
				stripped = stripped[len(prefix) : -len(suffix)].strip()
				changed = True
	return stripped


def _split_field_line(line: str) -> tuple[str | None, str]:
	match = re.match(
		r"^(?:\*\*|__|`)?([A-Za-z_][A-Za-z0-9_ ]*|[\u4e00-\u9fff]{1,10})(?:\*\*|__|`)?\s*[:：]\s*(.*)$",
		line,
	)
	if not match:
		return None, ""
	return match.group(1).strip(), match.group(2).strip()


def _split_first_colon(line: str) -> tuple[str, str]:
	left, _, right = line.partition(":")
	return left.strip(), right.strip()


def _strip_code_fences(text: str) -> str:
	stripped = text.strip()
	if stripped.startswith("```"):
		stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
		stripped = re.sub(r"\s*```$", "", stripped)
	return stripped.strip()