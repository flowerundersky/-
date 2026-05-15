"""Tests for prompt assembly."""

from src.prompt import PromptSpec, build_code_prompt_template, build_language_rules, build_self_check_prompt_template


def test_build_code_prompt_uses_stable_sections() -> None:
    spec = PromptSpec(
        task="Add a health check endpoint.",
        language="Python",
        context="Use FastAPI.",
        output_mode="code",
        style="minimal_change",
        constraints=("Keep the response minimal.",),
        references=("src/app.py",),
    )

    prompt = build_code_prompt_template(spec).invoke({})
    messages = prompt.to_messages()

    assert "You are a precise code generation assistant." in messages[0].content
    assert "Language Rules:" in messages[0].content
    assert "Add a health check endpoint." in messages[1].content
    assert "References:" in messages[1].content
    assert "src/app.py" in messages[1].content


def test_build_system_prompt_includes_language_rules_and_output_mode() -> None:
    spec = PromptSpec(task="", language="Python", output_mode="patch", style="preserve_structure")
    prompt = build_code_prompt_template(spec).invoke({})
    messages = prompt.to_messages()

    assert "Use snake_case for variables and functions." in messages[0].content
    assert "Return a unified diff or apply_patch-style patch." in messages[0].content
    assert "Preserve the current module and function structure where possible." in messages[0].content


def test_build_self_check_prompt_includes_draft_and_checklist() -> None:
    spec = PromptSpec(
        task="Add a health check endpoint.",
        language="Python",
        context="Use FastAPI.",
        output_mode="markdown",
        style="strict_format",
        constraints=("Keep the response minimal.",),
    )
    prompt = build_self_check_prompt_template(spec, "return {'ok': True}").invoke({})
    messages = prompt.to_messages()

    assert "你是一个严谨的代码审查助手。" in messages[0].content
    assert "请用中文汉字输出检查结论、下一步建议和补充说明。" in messages[0].content
    assert "return {'ok': True}" in messages[1].content
    assert "请只返回下面这些字段，字段值请使用中文汉字表达：" in messages[1].content


def test_build_revision_prompt_mentions_new_constraints() -> None:
    from src.prompt import build_revision_prompt_template

    spec = PromptSpec(
        task="Add a health check endpoint.",
        language="Python",
        context="Use FastAPI.",
        output_mode="code",
        style="minimal_change",
        constraints=("Keep the response minimal.",),
    )
    prompt = build_revision_prompt_template(
        spec,
        "return {'ok': True}",
        review_result="需要补充参数校验",
        user_instruction="新增约束：保留原有路由结构",
    ).invoke({})
    messages = prompt.to_messages()

    assert "请根据自检结果和用户在自检对话框里新增的约束修正草稿。" in messages[0].content
    assert "新增约束:" in messages[1].content
    assert "保留原有路由结构" in messages[1].content


def test_build_language_rules_uses_fallback_rules_for_unknown_language() -> None:
    prompt = build_language_rules("Rust")

    assert prompt == (
        "Match the project's existing style and naming conventions.",
        "Keep the solution focused and avoid unnecessary abstractions.",
    )