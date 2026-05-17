"""Gradio UI composition and event wiring."""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import lru_cache
from typing import Any

import gradio as gr

from .adapter import build_graph_request
from .graph import (
    GraphRequest,
    run_graph,
    run_graph_with_state,
)
from .model import QwenHFBackend, create_backend
from .parser import SMALL_SEMANTIC_LLM_MODEL_PATH, build_structured_fields_with_trace


logger = logging.getLogger(__name__)


def build_ui() -> gr.Blocks:
    """Build a natural-language driven workflow UI."""

    with gr.Blocks(title="Qwen Code Generation") as demo:
        gr.Markdown(
            "# Qwen Code Generation\n"
            "直接用自然语言描述任务、目标语言、输出形式、风格和是否自检。\n"
            "系统会自动抽取字段；自检模式会在图工作流内部完成 draft、review 和 revision 的循环，并输出最终自检摘要。"
        )
        chatbot = gr.Chatbot(label="对话", height=420)
        prompt_box = gr.Textbox(
            label="输入需求",
            lines=4,
            placeholder="例如：帮我改一下 parser，输出 patch，尽量少改；如果发现问题请自检",
        )
        self_check_box = gr.Checkbox(label="启用自检", value=False)
        revision_box = gr.Textbox(
            label="自检说明",
            lines=2,
            placeholder="图内自检已自动完成，这里仅保留为兼容界面。",
        )
        with gr.Row():
            preview_button = gr.Button("预览解析", variant="secondary")
            run_button = gr.Button("开始执行", variant="primary")
            continue_button = gr.Button("继续修正", variant="secondary")
            stop_button = gr.Button("结束自检", variant="secondary")
            clear_button = gr.Button("清空")

        semantic_preview = gr.Textbox(label="NLP模型字段块", lines=10)
        standardized_preview = gr.JSON(label="最终标准化结果")
        request_preview = gr.JSON(label="GraphRequest")
        result_box = gr.Textbox(label="输出结果", lines=14)
        conversation_state = gr.State([])
        workflow_state = gr.State({})

        prompt_box.submit(
            _handle_run,
            inputs=[prompt_box, self_check_box, conversation_state, workflow_state],
            outputs=[
                chatbot,
                semantic_preview,
                standardized_preview,
                request_preview,
                result_box,
                conversation_state,
                workflow_state,
                prompt_box,
                revision_box,
            ],
        )
        preview_button.click(
            _handle_preview,
            inputs=[prompt_box, self_check_box, conversation_state, workflow_state],
            outputs=[
                chatbot,
                semantic_preview,
                standardized_preview,
                request_preview,
                result_box,
                conversation_state,
                workflow_state,
                prompt_box,
                revision_box,
            ],
        )
        run_button.click(
            _handle_run,
            inputs=[prompt_box, self_check_box, conversation_state, workflow_state],
            outputs=[
                chatbot,
                semantic_preview,
                standardized_preview,
                request_preview,
                result_box,
                conversation_state,
                workflow_state,
                prompt_box,
                revision_box,
            ],
        )
        continue_button.click(
            _handle_continue,
            inputs=[revision_box, conversation_state, workflow_state],
            outputs=[
                chatbot,
                semantic_preview,
                standardized_preview,
                request_preview,
                result_box,
                conversation_state,
                workflow_state,
                prompt_box,
                revision_box,
            ],
        )
        stop_button.click(
            _handle_stop,
            inputs=[conversation_state, workflow_state],
            outputs=[
                chatbot,
                semantic_preview,
                standardized_preview,
                request_preview,
                result_box,
                conversation_state,
                workflow_state,
                prompt_box,
                revision_box,
            ],
        )
        clear_button.click(
            _handle_clear,
            outputs=[
                chatbot,
                semantic_preview,
                standardized_preview,
                request_preview,
                result_box,
                conversation_state,
                workflow_state,
                prompt_box,
                revision_box,
                self_check_box,
            ],
        )

    return demo


def _handle_preview(
    message: str,
    self_check: bool,
    history: list[list[str]],
    workflow_state: dict[str, Any],
) -> tuple[
    list[list[str]],
    str,
    dict[str, Any],
    dict[str, Any],
    str,
    list[list[str]],
    dict[str, Any],
    str,
    str,
]:
    try:
        trace, request = _prepare_request(message, self_check)
        assistant_text = _format_parsed_response(trace.final_payload)
    except Exception as exc:  # noqa: BLE001
        assistant_text = f"抽取失败：{exc}"
        updated_history = _append_chat(history, message, assistant_text)
        return (
            updated_history,
            assistant_text,
            {},
            {},
            assistant_text,
            updated_history,
            workflow_state,
            message,
            "",
        )

    updated_history = _append_chat(history, message, assistant_text)
    return (
        updated_history,
        _format_semantic_raw_output(trace.semantic_raw_output),
        trace.final_payload.model_dump(),
        _request_preview(request),
        assistant_text,
        updated_history,
        workflow_state,
        message,
        "",
    )


def _handle_run(
    message: str,
    self_check: bool,
    history: list[list[str]],
    workflow_state: dict[str, Any],
) -> tuple[
    list[list[str]],
    str,
    dict[str, Any],
    dict[str, Any],
    str,
    list[list[str]],
    dict[str, Any],
    str,
    str,
]:
    try:
        trace, request = _prepare_request(message, self_check)
    except Exception as exc:  # noqa: BLE001
        output = f"抽取失败：{exc}"
        updated_history = _append_chat(history, message, output)
        return (
            updated_history,
            output,
            {},
            {},
            output,
            updated_history,
            {},
            message,
            "",
        )

    try:
        if request.self_check:
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
                backend=_get_generation_backend(),
            )
            output = str(graph_state.get("output", ""))
            updated_workflow_state = _serialize_graph_state(graph_state, request, message)
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
                backend=_get_generation_backend(),
            )
            updated_workflow_state = {}
    except Exception as exc:  # noqa: BLE001
        output = f"生成失败：{exc}"
        updated_workflow_state = {}

    updated_history = _append_chat(history, message, output)
    return (
        updated_history,
        _format_semantic_raw_output(trace.semantic_raw_output),
        trace.final_payload.model_dump(),
        _request_preview(request),
        output,
        updated_history,
        updated_workflow_state,
        message,
        "",
    )


def _handle_continue(
    revision_note: str,
    history: list[list[str]],
    workflow_state: dict[str, Any],
) -> tuple[
    list[list[str]],
    str,
    dict[str, Any],
    dict[str, Any],
    str,
    list[list[str]],
    dict[str, Any],
    str,
    str,
]:
    request_data = workflow_state.get("request") or {}
    draft_code = str(workflow_state.get("draft_code", "")).strip()
    review_result = str(workflow_state.get("review_result", "")).strip()
    if not request_data or not draft_code or not review_result:
        output = "当前没有可继续修订的自检结果，请先点击‘开始执行’。"
        updated_history = _append_chat(history, "继续修正", output)
        return (
            updated_history,
            output,
            {},
            {},
            output,
            updated_history,
            workflow_state,
            "",
            "",
        )

    try:
        request = GraphRequest(
            task=str(request_data.get("task", "")),
            language=str(request_data.get("language", "")),
            context=str(request_data.get("context", "")),
            output_mode=request_data.get("output_mode", "code"),
            style=request_data.get("style", "minimal_change"),
            constraints=tuple(request_data.get("constraints", ())),
            references=tuple(request_data.get("references", ())),
            draft_code=draft_code,
            self_check=True,
            review_result=review_result,
            user_instruction=revision_note.strip(),
            generation_params=request_data.get("generation_params"),
        )
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
            backend=_get_generation_backend(),
        )
        output = str(graph_state.get("output", ""))
        updated_workflow_state = _serialize_graph_state(graph_state, request, workflow_state.get("source_message", ""))
    except Exception as exc:  # noqa: BLE001
        output = f"继续修正失败：{exc}"
        updated_workflow_state = workflow_state

    updated_history = _append_chat(history, "继续修正", output)
    return (
        updated_history,
        output,
        {},
        {},
        output,
        updated_history,
        {},
        "",
        "",
    )


def _handle_stop(
    history: list[list[str]],
    workflow_state: dict[str, Any],
) -> tuple[
    list[list[str]],
    str,
    dict[str, Any],
    dict[str, Any],
    str,
    list[list[str]],
    dict[str, Any],
    str,
    str,
]:
    _ = workflow_state
    output = "自检已内置到图工作流中，当前没有可单独结束的外部会话。"
    updated_history = _append_chat(history, "结束自检", output)
    return (
        updated_history,
        output,
        {},
        {},
        output,
        updated_history,
        {},
        "",
        "",
    )


def _handle_clear() -> tuple[
    list[list[str]],
    str,
    dict[str, Any],
    dict[str, Any],
    str,
    list[list[str]],
    dict[str, Any],
    str,
    str,
    bool,
]:
    return ([], "", {}, {}, "", [], {}, "", "", False)


def _prepare_request(message: str, self_check: bool) -> tuple[Any, GraphRequest]:
    full_text = message.strip()
    logger.debug(
        "ui parse input received: message_len=%s message_preview=%r",
        len(full_text),
        full_text[:200].replace("\n", "\\n"),
    )
    trace = build_structured_fields_with_trace(full_text, semantic_backend=_get_semantic_backend())

    if not trace.semantic_raw_output:
        raise RuntimeError(f"抽取失败：输入文本未得到任何小模型输出 input_preview={full_text[:200]!r}")

    logger.info(
        "semantic extraction produced raw output: raw_output_len=%s parsed_keys=%s",
        len(trace.semantic_raw_output),
        sorted(trace.semantic_fields.keys()),
    )

    request = build_graph_request(trace.final_payload)
    request = replace(request, self_check=self_check)
    return trace, request


def _format_parsed_response(parsed: Any) -> str:
    return "已解析字段：\n" + _format_field_block(parsed.model_dump())


def _format_field_block(fields: dict[str, Any]) -> str:
    if not fields:
        return "未抽取到字段。"

    lines: list[str] = []
    for key, value in fields.items():
        lines.extend(_format_field_lines(key, value))
    return "\n".join(lines)


def _format_field_lines(key: str, value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{key}: {{}}"]
        lines = [f"{prefix}{key}:"]
        for inner_key, inner_value in value.items():
            lines.extend(_format_field_lines(inner_key, inner_value, indent + 2))
        return lines

    if isinstance(value, (list, tuple)):
        if not value:
            return [f"{prefix}{key}: []"]
        lines = [f"{prefix}{key}:"]
        for item in value:
            if isinstance(item, dict):
                lines.extend(_format_field_lines("-", item, indent + 2))
            else:
                lines.append(f"{prefix}  - {item}")
        return lines

    if value in (None, ""):
        return [f"{prefix}{key}: "]
    return [f"{prefix}{key}: {value}"]


def _append_chat(history: list[dict[str, str]], user_text: str, assistant_text: str) -> list[dict[str, str]]:
    updated = list(history or [])
    updated.append({"role": "user", "content": user_text})
    updated.append({"role": "assistant", "content": assistant_text})
    return updated


def _format_semantic_raw_output(raw_output: str) -> str:
    text = (raw_output or "").strip()
    if not text:
        return "未抽取到小模型原始输出。"
    return text


def _request_preview(request: GraphRequest) -> dict[str, Any]:
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


def _serialize_graph_state(graph_state: dict[str, Any], request: GraphRequest, source_message: str = "") -> dict[str, Any]:
    review = graph_state.get("review")
    serialized_review: dict[str, Any] = {}
    if review is not None:
        serialized_review = {
            "review_result": getattr(review, "review_result", ""),
            "next_step": getattr(review, "next_step", ""),
            "recommend_continue": bool(getattr(review, "recommend_continue", False)),
            "notes": getattr(review, "notes", ""),
            "raw_output": getattr(review, "raw_output", ""),
        }

    return {
        "kind": "graph_state",
        "source_message": source_message,
        "request": _request_preview(request),
        "draft_code": str(graph_state.get("draft_code", "")),
        "review_result": str(graph_state.get("review_result", serialized_review.get("review_result", ""))),
        "review": serialized_review,
        "round_index": int(graph_state.get("round_index", 0)),
        "max_rounds": int(graph_state.get("max_rounds", 2)),
        "output": str(graph_state.get("output", "")),
    }


@lru_cache(maxsize=1)
def _get_semantic_backend() -> QwenHFBackend:
    return QwenHFBackend(model_path=SMALL_SEMANTIC_LLM_MODEL_PATH).load()


@lru_cache(maxsize=1)
def _get_generation_backend():
    return create_backend().load()
