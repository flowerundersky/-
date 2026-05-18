#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))


def main() -> int:
	from src.adapter import build_graph_request
	from src.graph import run_graph, run_graph_with_state
	from src.parser import build_structured_fields_with_trace

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

	if not message:
		print(json.dumps({"ok": False, "error": "message cannot be empty"}, ensure_ascii=False))
		return 1

	try:
		trace = build_structured_fields_with_trace(message)
		request = build_graph_request(trace.final_payload)
		request = replace(request, self_check=self_check)
		request_payload = _serialize_request(request)

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
			output = str(graph_state.get("output", ""))
			review = graph_state.get("review")
			result = {
				"ok": True,
				"mode": mode,
				"message": message,
				"self_check": self_check,
				"trace": trace.final_payload.model_dump(),
				"request": request_payload,
				"output": output,
				"review": {
					"review_result": getattr(review, "review_result", ""),
					"next_step": getattr(review, "next_step", ""),
					"recommend_continue": bool(getattr(review, "recommend_continue", False)),
					"notes": getattr(review, "notes", ""),
					"raw_output": getattr(review, "raw_output", ""),
				},
			}
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


def _serialize_request(request) -> dict[str, object]:
	return {
		"task": request.task,
		"language": request.language,
		"context": request.context,
		"output_mode": request.output_mode,
		"style": request.style,
		"constraints": list(request.constraints),
		"references": list(request.references),
		"draft_code": request.draft_code,
		"self_check": request.self_check,
		"review_result": request.review_result,
		"user_instruction": request.user_instruction,
		"generation_params": request.generation_params,
	}


if __name__ == "__main__":
	raise SystemExit(main())
