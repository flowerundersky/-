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
    ReviewOutcome,
    SelfCheckSession,
    continue_self_check_session,
    format_review_message,
    run_graph,
    start_self_check_session,
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
            "系统会自动抽取字段；自检模式会进入显式 review 状态机，等待你补充新的约束并决定是否继续。"
        )
        chatbot = gr.Chatbot(label="对话", height=420)
        prompt_box = gr.Textbox(
            label="输入需求",
            lines=4,
            placeholder="例如：帮我改一下 parser，输出 patch，尽量少改；如果发现问题请自检",
        )
        self_check_box = gr.Checkbox(label="启用自检", value=False)
        revision_box = gr.Textbox(
            label="自检新增约束（可选）",
            lines=2,
            placeholder="例如：请优先修复字段抽取的歧义，保留当前结构",
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
            session = start_self_check_session(request, backend=_get_generation_backend())
            output = _decorate_review_message(format_review_message(session), session)
            updated_workflow_state = _serialize_session(session, source_message=message)
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
                self_check=False,
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
    session = _deserialize_session(workflow_state)
    if session is None or not session.request.self_check:
        output = "当前没有可继续的自检会话。"
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
        updated_session = continue_self_check_session(
            session,
            backend=_get_generation_backend(),
            user_instruction=revision_note,
        )
        output = _decorate_review_message(format_review_message(updated_session), updated_session)
        updated_workflow_state = _serialize_session(
            updated_session,
            source_message=workflow_state.get("source_message", ""),
        )
    except Exception as exc:  # noqa: BLE001
        output = f"继续修正失败：{exc}"
        updated_workflow_state = workflow_state

    updated_history = _append_chat(history, "继续修正", output)
    return (
        updated_history,
        _format_semantic_raw_output(""),
        {},
        _request_preview(session.request),
        output,
        updated_history,
        updated_workflow_state,
        str(workflow_state.get("source_message", "")),
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
    session = _deserialize_session(workflow_state)
    if session is None:
        output = "当前没有需要结束的自检会话。"
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

    output = (
        "已结束自检会话。\n\n"
        f"当前草稿：\n{session.draft_code.strip() or '无'}\n\n"
        f"最近一次检查结果：\n{session.review.review_result.strip() or '无'}"
    )
    updated_history = _append_chat(history, "结束自检", output)
    return (
        updated_history,
        _format_semantic_raw_output(""),
        {},
        _request_preview(session.request),
        output,
        updated_history,
        {},
        str(workflow_state.get("source_message", "")),
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


def _decorate_review_message(message: str, session: SelfCheckSession) -> str:
    recommendation = "建议继续" if session.review.recommend_continue else "可以结束"
    max_rounds = f"{session.round_index + 1}/{session.max_rounds}"
    return (
        f"{message}\n\n"
        f"当前轮次：{max_rounds}\n"
        "你可以点击“继续修正”进入下一轮，或者点击“结束自检”结束当前会话。\n"
        f"模型建议：{recommendation}"
    )


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
        "generation_params": request.generation_params,
    }


def _serialize_session(session: SelfCheckSession, source_message: str = "") -> dict[str, Any]:
    return {
        "kind": "self_check_session",
        "source_message": source_message,
        "request": _request_preview(session.request),
        "draft_code": session.draft_code,
        "review": {
            "review_result": session.review.review_result,
            "next_step": session.review.next_step,
            "recommend_continue": session.review.recommend_continue,
            "notes": session.review.notes,
            "raw_output": session.review.raw_output,
        },
        "round_index": session.round_index,
        "max_rounds": session.max_rounds,
    }


def _deserialize_session(data: dict[str, Any]) -> SelfCheckSession | None:
    if not data or data.get("kind") != "self_check_session":
        return None

    request_data = data.get("request") or {}
    review_data = data.get("review") or {}
    request = GraphRequest(
        task=request_data.get("task", ""),
        language=request_data.get("language", ""),
        context=request_data.get("context", ""),
        output_mode=request_data.get("output_mode", "code"),
        style=request_data.get("style", "minimal_change"),
        constraints=tuple(request_data.get("constraints", ())),
        references=tuple(request_data.get("references", ())),
        draft_code=request_data.get("draft_code"),
        self_check=bool(request_data.get("self_check", False)),
        generation_params=request_data.get("generation_params"),
    )
    review = {
        "review_result": review_data.get("review_result", ""),
        "next_step": review_data.get("next_step", ""),
        "recommend_continue": bool(review_data.get("recommend_continue", False)),
        "notes": review_data.get("notes", ""),
        "raw_output": review_data.get("raw_output", ""),
    }
    return SelfCheckSession(
        request=request,
        draft_code=data.get("draft_code", ""),
        review=ReviewOutcome(
            review_result=review["review_result"],
            next_step=review["next_step"],
            recommend_continue=review["recommend_continue"],
            notes=review["notes"],
            raw_output=review["raw_output"],
        ),
        round_index=int(data.get("round_index", 0)),
        max_rounds=int(data.get("max_rounds", 2)),
    )


@lru_cache(maxsize=1)
def _get_semantic_backend() -> QwenHFBackend:
    return QwenHFBackend(model_path=SMALL_SEMANTIC_LLM_MODEL_PATH).load()


@lru_cache(maxsize=1)
def _get_generation_backend():
    return create_backend().load()
