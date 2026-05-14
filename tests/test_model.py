"""Tests for model utilities."""

import torch

from src.model import GenerationParams, QwenHFBackend, _prompt_to_model_inputs, _truncate_by_stop_sequences


class DummyPrompt:
    def to_messages(self):
        return [
            {"role": "system", "content": "system note"},
            {"role": "user", "content": "hello"},
        ]


class DummyMessage:
    def __init__(self, message_type: str, content: str) -> None:
        self.type = message_type
        self.content = content


class DummyObjectPrompt:
    def to_messages(self):
        return [
            DummyMessage("system", "system note"),
            DummyMessage("human", "hello"),
            DummyMessage("ai", "previous reply"),
        ]


class DummyTokenizer:
    def __init__(self) -> None:
        self.seen_messages = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, return_tensors=None):
        self.seen_messages = list(messages)
        if not tokenize:
            return "system note\nhello\nassistant"
        assert add_generation_prompt is True
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

    def __call__(self, *args, **kwargs):
        raise AssertionError("fallback tokenizer path should not be used")


def test_generation_params_merge_stop_sequences() -> None:
    params = GenerationParams.from_input(
        {
            "max_new_tokens": 32,
            "min_new_tokens": 8,
            "temperature": 0.7,
            "stop_words": ["END"],
            "stop_sequences": ["STOP", "END"],
            "repetition_penalty": 1.2,
            "extra_flag": True,
        }
    )

    assert params.max_new_tokens == 32
    assert params.min_new_tokens == 8
    assert params.temperature == 0.7
    assert params.repetition_penalty == 1.2
    assert params.combined_stop_sequences == ("STOP", "END")
    assert params.extra_kwargs == {"extra_flag": True}


def test_truncate_by_stop_sequences() -> None:
    assert _truncate_by_stop_sequences("hello world STOP after", ["STOP"]) == "hello world"


def test_prompt_to_model_inputs_uses_chat_template_path() -> None:
    tokenizer = DummyTokenizer()
    inputs = _prompt_to_model_inputs(DummyPrompt(), tokenizer)

    assert tokenizer.seen_messages == DummyPrompt().to_messages()
    assert inputs["input_ids"].shape == (1, 3)
    assert inputs["attention_mask"].shape == (1, 3)


def test_prompt_to_model_inputs_normalizes_object_messages() -> None:
    tokenizer = DummyTokenizer()

    _prompt_to_model_inputs(DummyObjectPrompt(), tokenizer)

    assert tokenizer.seen_messages == [
        {"role": "system", "content": "system note"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "previous reply"},
    ]


def test_backend_factory_returns_qwen_backend() -> None:
    backend = QwenHFBackend()

    assert isinstance(backend, QwenHFBackend)