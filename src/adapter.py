"""Pydantic-based adapter for turning structured fields into graph requests."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .graph import GraphRequest
from .parser import ParsedRequestPayload
from .prompt import OutputMode, StyleMode


class GraphRequestPayload(BaseModel):
	"""Validated structured payload that is ready to become a graph request."""

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
			return ()
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
			return ""

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
	def _validate_self_check(self) -> "GraphRequestPayload":
		return self

	def to_graph_request(self) -> GraphRequest:
		"""Convert the validated structured payload into a graph request."""

		return GraphRequest(
			task=self.task,
			language=self.language,
			context=self.context,
			output_mode=self.output_mode,
			style=self.style,
			constraints=tuple(self.constraints),
			references=tuple(self.references),
			draft_code=self.draft_code,
			self_check=self.self_check,
			generation_params=self.generation_params or None,
		)


def build_graph_request(
	payload: Mapping[str, Any] | ParsedRequestPayload | GraphRequest | GraphRequestPayload,
) -> GraphRequest:
	"""Validate structured fields and normalize them for the graph layer."""

	if isinstance(payload, GraphRequest):
		return payload

	if isinstance(payload, GraphRequestPayload):
		return payload.to_graph_request()

	if isinstance(payload, ParsedRequestPayload):
		return GraphRequestPayload.model_validate(payload.model_dump()).to_graph_request()

	return GraphRequestPayload.model_validate(payload).to_graph_request()
