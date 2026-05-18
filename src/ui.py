"""Gradio UI composition and event wiring."""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import gradio as gr


logger = logging.getLogger(__name__)
GO_API_BASE_URL = os.getenv("GO_API_BASE_URL", "http://127.0.0.1:8088").rstrip("/")


def build_ui() -> gr.Blocks:
    """Build a natural-language driven workflow UI."""

    css = """
    .app-shell {
        max-width: 1280px;
        margin: 0 auto;
        padding: 24px 16px 32px;
    }
    .hero {
        border-radius: 24px;
        padding: 28px 28px 22px;
        background: linear-gradient(135deg, rgba(18, 24, 38, 0.98), rgba(33, 51, 87, 0.92));
        color: #f6f7fb;
        box-shadow: 0 16px 48px rgba(11, 18, 32, 0.18);
        margin-bottom: 18px;
    }
    .hero h1 {
        font-size: 32px;
        line-height: 1.1;
        margin: 0 0 10px 0;
    }
    .hero p {
        margin: 0;
        color: rgba(246, 247, 251, 0.82);
        max-width: 820px;
        font-size: 15px;
    }
    .hero-pills {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 18px;
    }
    .hero-pill {
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #eef2ff;
        font-size: 12px;
        letter-spacing: 0.02em;
    }
    .panel {
        border-radius: 20px;
        border: 1px solid rgba(31, 41, 55, 0.08);
        background: rgba(255, 255, 255, 0.96);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        padding: 18px;
    }
    .panel-title {
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #0f172a;
    }
    .panel-subtitle {
        font-size: 13px;
        color: #475569;
        margin-bottom: 14px;
    }
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 18px 0;
    }
    .stat-card {
        border-radius: 16px;
        padding: 14px 16px;
        background: linear-gradient(180deg, #ffffff, #f8fafc);
        border: 1px solid #e2e8f0;
    }
    .stat-card .label {
        display: block;
        font-size: 12px;
        color: #64748b;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .stat-card .value {
        display: block;
        font-size: 16px;
        color: #0f172a;
        font-weight: 700;
    }
    .output-tabs {
        margin-top: 8px;
    }
    """

    with gr.Blocks(title="Qwen Code Generation", theme=gr.themes.Soft(), css=css) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <div class="hero">
                  <h1>Qwen Code Generation</h1>
                  <p>浏览器里的前端负责交互，Go 网关负责公网治理，Python agent 负责解析、编排和生成。按钮按下后，流程会沿着前端 -> Go -> agent 的路径执行。</p>
                  <div class="hero-pills">
                    <span class="hero-pill">UI as Frontend</span>
                    <span class="hero-pill">Go Gateway</span>
                    <span class="hero-pill">Python Agent</span>
                    <span class="hero-pill">Preview / Generate / Continue</span>
                  </div>
                </div>
                """
            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=5, min_width=360):
                    gr.HTML(
                        """
                        <div class="panel">
                          <div class="panel-title">请求控制台</div>
                          <div class="panel-subtitle">输入自然语言需求，选择是否启用自检，然后先预览或直接执行。</div>
                        </div>
                        """
                    )
                    prompt_box = gr.Textbox(
                        label="输入需求",
                        lines=5,
                        placeholder="例如：帮我改一下 parser，输出 patch，尽量少改；如果发现问题请自检",
                    )
                    self_check_box = gr.Checkbox(label="启用自检", value=False)
                    revision_box = gr.Textbox(
                        label="继续修正说明",
                        lines=3,
                        placeholder="如果已有自检结果，这里输入新的约束或修订意见。",
                    )
                    with gr.Row():
                        preview_button = gr.Button("预览解析", variant="secondary")
                        run_button = gr.Button("开始执行", variant="primary")
                    with gr.Row():
                        continue_button = gr.Button("继续修正", variant="secondary")
                        stop_button = gr.Button("结束自检", variant="secondary")
                        clear_button = gr.Button("清空")

                    gr.HTML(
                        """
                        <div class="stats-grid">
                          <div class="stat-card"><span class="label">入口</span><span class="value">前端 -> Go</span></div>
                          <div class="stat-card"><span class="label">处理</span><span class="value">Go -> Agent</span></div>
                          <div class="stat-card"><span class="label">输出</span><span class="value">生成结果</span></div>
                        </div>
                        """
                    )

                with gr.Column(scale=7, min_width=420):
                    gr.HTML(
                        """
                        <div class="panel">
                          <div class="panel-title">结果工作区</div>
                          <div class="panel-subtitle">左侧提交请求后，这里展示对话、生成结果和结构化预览。</div>
                        </div>
                        """
                    )
                    chatbot = gr.Chatbot(label="对话", height=360)
                    result_box = gr.Textbox(label="输出结果", lines=12)
                    with gr.Tabs(elem_classes=["output-tabs"]):
                        with gr.Tab("结构化预览"):
                            semantic_preview = gr.Textbox(label="NLP 模型字段块", lines=10)
                            standardized_preview = gr.JSON(label="最终标准化结果")
                            request_preview = gr.JSON(label="GraphRequest")
                        with gr.Tab("结果说明"):
                            gr.Markdown(
                                "- 预览解析：显示 Go 网关返回的结构化字段。\n"
                                "- 开始执行：显示 agent 生成结果。\n"
                                "- 继续修正：沿用当前自检状态继续下一轮。"
                            )

            conversation_state = gr.State([])
            workflow_state = gr.State({})

            common_outputs = [
                chatbot,
                semantic_preview,
                standardized_preview,
                request_preview,
                result_box,
                conversation_state,
                workflow_state,
                prompt_box,
                revision_box,
            ]

            prompt_box.submit(_handle_run, inputs=[prompt_box, self_check_box, conversation_state, workflow_state], outputs=common_outputs)
            preview_button.click(_handle_preview, inputs=[prompt_box, self_check_box, conversation_state, workflow_state], outputs=common_outputs)
            run_button.click(_handle_run, inputs=[prompt_box, self_check_box, conversation_state, workflow_state], outputs=common_outputs)
            continue_button.click(_handle_continue, inputs=[revision_box, conversation_state, workflow_state], outputs=common_outputs)
            stop_button.click(_handle_stop, inputs=[conversation_state, workflow_state], outputs=common_outputs)
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
        response = _call_go_api("/v1/preview", {"message": message, "self_check": self_check})
        result = _unwrap_result(response)
        trace = result.get("trace") or {}
        request = result.get("request") or {}
        assistant_text = _format_parsed_response(trace)
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
        _format_parsed_response(trace),
        trace,
        request,
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
        response = _call_go_api("/v1/generate", {"message": message, "self_check": self_check})
        result = _unwrap_result(response)
        trace = result.get("trace") or {}
        request = result.get("request") or {}
        output = str(result.get("output", ""))
        updated_workflow_state = result if isinstance(result.get("review"), dict) else {}
    except Exception as exc:  # noqa: BLE001
        output = f"生成失败：{exc}"
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

    updated_history = _append_chat(history, message, output)
    return (
        updated_history,
        _format_parsed_response(trace),
        trace,
        request,
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
    if not workflow_state or not workflow_state.get("review") or not workflow_state.get("request"):
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
        response = _call_go_api(
            "/v1/generate",
            {"revision_note": revision_note, "workflow_state": workflow_state},
        )
        result = _unwrap_result(response)
        trace = result.get("trace") or {}
        request = result.get("request") or {}
        output = str(result.get("output", ""))
        updated_workflow_state = result if isinstance(result.get("review"), dict) else workflow_state
    except Exception as exc:  # noqa: BLE001
        output = f"继续修正失败：{exc}"
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

    updated_history = _append_chat(history, "继续修正", output)
    return (
        updated_history,
        _format_parsed_response(trace),
        trace,
        request,
        output,
        updated_history,
        updated_workflow_state,
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
    output = "自检流程由 Go 承接层统一管理，当前只清理前端状态。"
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


def _call_go_api(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{GO_API_BASE_URL}{path}"
    request = urllib_request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=300) as response:
            body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(_extract_error_message(error_body) or f"Go API 请求失败: HTTP {exc.code}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"无法连接 Go API: {exc.reason}") from exc

    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Go API 返回了非 JSON 内容: {exc}") from exc

    return decoded


def _unwrap_result(response: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise RuntimeError("Go API 响应格式错误")
    if not response.get("ok"):
        raise RuntimeError(_extract_error_message(response) or "Go API 返回失败")

    result = response.get("result") or {}
    if not isinstance(result, dict):
        raise RuntimeError("Go API result 字段格式错误")
    if not result.get("ok", True):
        raise RuntimeError(_extract_error_message(result) or "Go API 业务返回失败")
    return result


def _format_parsed_response(parsed: dict[str, Any]) -> str:
    return "已解析字段：\n" + _format_field_block(parsed)


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


def _extract_error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(payload, str) and payload.strip():
        return payload.strip()
    return ""
