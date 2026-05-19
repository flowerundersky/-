#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from contextlib import redirect_stdout
from src.graph import GraphRequest,run_graph_with_state

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))


def main() -> int:
	raw = sys.stdin.read()
	if not raw.strip():
		print(json.dumps({"ok": False, "error": "empty request body"}, ensure_ascii=False))
		return 1

	try:
		payload = json.loads(raw)
	except json.JSONDecodeError as exc:
		print(json.dumps({"ok": False, "error": f"invalid json: {exc}"}, ensure_ascii=False))
		return 1

	message = str(payload.get("message", "")).strip()
	mode = str(payload.get("mode", "generate")).strip() or "generate"
	self_check = bool(payload.get("self_check", False))
	revision_note = str(payload.get("revision_note", "")).strip()
	workflow_state = payload.get("workflow_state") or {}

	if mode in {"preview", "generate"} and not message:
		print(json.dumps({"ok": False, "error": "message cannot be empty"}, ensure_ascii=False))
		return 1
	if mode == "continue":
		if not revision_note:
			print(json.dumps({"ok": False, "error": "revision_note cannot be empty"}, ensure_ascii=False))
			return 1
		if not isinstance(workflow_state, dict) or not workflow_state:
			print(json.dumps({"ok": False, "error": "workflow_state cannot be empty"}, ensure_ascii=False))
			return 1

	try:
		with redirect_stdout(sys.stderr):
			from src.adapter import build_graph_request
			from src.graph import run_graph, run_graph_with_state
			from src.parser import build_structured_fields_with_trace

			if mode == "continue":
				result = _run_continue(workflow_state, revision_note)
				print(json.dumps(result, ensure_ascii=False))
				return 0

			trace = build_structured_fields_with_trace(message)
			request = build_graph_request(trace.final_payload)
			request_payload = _serialize_request(request, self_check=self_check)

			if mode == "preview":
				result = {
					"ok": True,
					"mode": mode,
					"message": message,
					"self_check": self_check,
					"trace": trace.final_payload.model_dump(),
					"request": request_payload,
				}
				print(json.dumps(result, ensure_ascii=False))
				return 0

			if self_check:
				graph_state = run_graph_with_state(
					request.task,
					request.language,
					request.context,
					output_mode=request.output_mode,
					style=request.style,
					constraints=request.constraints,
					references=request.references,
					draft_code=request.draft_code,
					self_check=request.self_check,
					review_result="",
					user_instruction="",
					generation_params=request.generation_params,
				)
				result = _serialize_graph_state(graph_state, request_payload, trace.final_payload.model_dump(), message, mode=mode)
			else:
				output = run_graph(
					request.task,
					request.language,
					request.context,
					output_mode=request.output_mode,
					style=request.style,
					constraints=request.constraints,
					references=request.references,
					draft_code=request.draft_code,
					self_check=request.self_check,
					generation_params=request.generation_params,
				)
				result = {
					"ok": True,
					"mode": mode,
					"message": message,
					"self_check": self_check,
					"trace": trace.final_payload.model_dump(),
					"request": request_payload,
					"output": output,
				}
		print(json.dumps(result, ensure_ascii=False))
		return 0
	except Exception as exc:  # noqa: BLE001
		print(json.dumps({"ok": False, "mode": mode, "error": str(exc)}, ensure_ascii=False))
		return 1


def _serialize_request(request, self_check: bool = False) -> dict[str, object]:
	return {
		"task": request.task,
		"language": request.language,
		"context": request.context,
		"output_mode": request.output_mode,
		"style": request.style,
		"constraints": list(request.constraints),
		"references": list(request.references),
		"draft_code": request.draft_code,
		"self_check": self_check,
		"review_result": request.review_result,
		"user_instruction": request.user_instruction,
		"generation_params": request.generation_params,
	}


def _run_continue(workflow_state: dict[str, object], revision_note: str) -> dict[str, object]:
	request_data = workflow_state.get("request") or {}
	if not isinstance(request_data, dict):
		raise ValueError("workflow_state request must be a mapping")



	request = GraphRequest(
		task=str(request_data.get("task", "")),
		language=str(request_data.get("language", "")),
		context=str(request_data.get("context", "")),
		output_mode=request_data.get("output_mode", "code"),
		style=request_data.get("style", "minimal_change"),
		constraints=tuple(request_data.get("constraints", ())),
		references=tuple(request_data.get("references", ())),
		draft_code=str(workflow_state.get("draft_code", "")) or None,
		self_check=True,
		review_result=str(workflow_state.get("review_result", "")),
		user_instruction=revision_note.strip(),
		generation_params=request_data.get("generation_params"),
	)
	request_payload = _serialize_request(request, self_check=True)
	graph_state = run_graph_with_state(
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
		generation_params=request.generation_params,
	)
	return _serialize_graph_state(graph_state, request_payload, request_payload, revision_note)


def _serialize_graph_state(
	graph_state: dict[str, object],
	request: dict[str, object],
	trace: dict[str, object],
	source_message: str = "",
	*,
	mode: str = "continue",
) -> dict[str, object]:
	review = graph_state.get("review")
	serialized_review: dict[str, object] = {}
	if review is not None:
		serialized_review = {
			"review_result": getattr(review, "review_result", ""),
			"next_step": getattr(review, "next_step", ""),
			"recommend_continue": bool(getattr(review, "recommend_continue", False)),
			"notes": getattr(review, "notes", ""),
			"raw_output": getattr(review, "raw_output", ""),
		}

	return {
		"ok": True,
		"mode": mode,
		"message": source_message,
		"self_check": True,
		"trace": trace,
		"request": request,
		"draft_code": str(graph_state.get("draft_code", "")),
		"review_result": str(graph_state.get("review_result", serialized_review.get("review_result", ""))),
		"review": serialized_review,
		"round_index": int(graph_state.get("round_index", 0)),
		"max_rounds": int(graph_state.get("max_rounds", 2)),
		"output": str(graph_state.get("output", "")),
	}


if __name__ == "__main__":
	raise SystemExit(main())
