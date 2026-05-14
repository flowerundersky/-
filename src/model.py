"""Model loading and generation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import gc
import logging

import torch
from langchain_core.prompt_values import PromptValue
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import config


BackendName = str


logger = logging.getLogger(__name__)


class ModelBackend(Protocol):
    """Common interface for model backends."""

    def load(self) -> "ModelBackend":
        """Load backend resources and return the backend instance."""

    def generate(self, prompt: Any, params: Mapping[str, Any] | None = None) -> str:
        """Generate plain text from a LangChain prompt value and generation parameters."""

    def close(self) -> None:
        """Release backend resources."""


@dataclass(slots=True)
class GenerationParams:
    """Normalized generation inputs shared by backends."""

    max_new_tokens: int = config.DEFAULT_MAX_NEW_TOKENS
    min_new_tokens: int = config.DEFAULT_MIN_NEW_TOKENS
    temperature: float = config.DEFAULT_TEMPERATURE
    top_p: float = config.DEFAULT_TOP_P
    repetition_penalty: float = config.DEFAULT_REPETITION_PENALTY
    
    stop_sequences: Sequence[str] = field(default_factory=tuple)
    stop_words: Sequence[str] = field(default_factory=tuple)
    do_sample: bool | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_input(cls, params: Mapping[str, Any] | None) -> "GenerationParams":
        if params is None:
            return cls()

        known_keys = {
            "max_new_tokens",
            "min_new_tokens",
            "temperature",
            "top_p",
            "repetition_penalty",
            "stop_sequences",
            "stop_words",
            "do_sample",
        }
        extra_kwargs = {key: value for key, value in params.items() if key not in known_keys}
        return cls(
            max_new_tokens=int(params.get("max_new_tokens", config.DEFAULT_MAX_NEW_TOKENS)),
            min_new_tokens=int(params.get("min_new_tokens", config.DEFAULT_MIN_NEW_TOKENS)),
            temperature=float(params.get("temperature", config.DEFAULT_TEMPERATURE)),
            top_p=float(params.get("top_p", config.DEFAULT_TOP_P)),
            repetition_penalty=float(
                params.get("repetition_penalty", config.DEFAULT_REPETITION_PENALTY)
            ),
            stop_sequences=_coerce_stop_sequences(params.get("stop_sequences")),
            stop_words=_coerce_stop_sequences(params.get("stop_words")),
            do_sample=params.get("do_sample"),
            extra_kwargs=extra_kwargs,
        )

    @property
    def combined_stop_sequences(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.stop_sequences, *self.stop_words)))

    def to_generation_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample
            if self.do_sample is not None
            else self.temperature > 0 or self.top_p < 1,
        }
        kwargs.update(self.extra_kwargs)
        return kwargs

@dataclass(slots=True)
class QwenHFBackend:
    """Transformers backend for local Qwen models."""

    model_name: str = config.DEFAULT_MODEL_NAME
    model_path: str = config.DEFAULT_MODEL_PATH
    device: str = config.DEFAULT_DEVICE
    tokenizer: Any | None = field(default=None, init=False, repr=False)
    model: Any | None = field(default=None, init=False, repr=False)
    _device_type: str | None = field(default=None, init=False, repr=False)

    def load(self) -> "QwenHFBackend":
        model_source = self.model_path or self.model_name
        torch_dtype, device_map, device_type = _resolve_runtime_settings(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        _ensure_chat_template(self.tokenizer)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=torch_dtype,
            device_map=device_map,
            local_files_only=True,
        )
        if device_type == "cpu":
            self.model = self.model.to(torch.device("cpu"))

        self._device_type = device_type
        return self

    def generate(self, prompt: Any, params: Mapping[str, Any] | None = None) -> str:
        if self.model is None or self.tokenizer is None:
            self.load()

        generation_params = GenerationParams.from_input(params)



        inputs = _prompt_to_model_inputs(prompt, self.tokenizer)
        input_ids = inputs.get("input_ids")
        logger.debug(
            "model tokenizer output: input_ids_shape=%s attention_mask_shape=%s",
            tuple(input_ids.shape) if input_ids is not None else None,
            tuple(inputs["attention_mask"].shape) if "attention_mask" in inputs else None,
        )
        inputs = _move_batch_to_device(inputs, self.model, self._device_type or "cpu")

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **generation_params.to_generation_kwargs(),
            )

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return _truncate_by_stop_sequences(text, generation_params.combined_stop_sequences)

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self._device_type = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _prompt_to_model_inputs(prompt: Any, tokenizer: Any) -> dict[str, Any]:
    messages = _prompt_to_messages(prompt)

    logger.debug("model prompt messages: %s", _format_prompt_messages(messages))

    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    logger.debug("model prompt rendered: %s", str(rendered_prompt)[:400].replace("\n", "\\n"))

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if isinstance(inputs, torch.Tensor):
        return {"input_ids": inputs}
    if isinstance(inputs, Mapping):
        return dict(inputs)
    return {"input_ids": inputs}


def _ensure_chat_template(tokenizer: Any) -> None:
    if getattr(tokenizer, "chat_template", None):
        return

    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% elif message['role'] in ['user', 'human'] %}"
        "{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% elif message['role'] in ['assistant', 'ai'] %}"
        "{{ '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )


def create_backend(backend_name: BackendName | None = None) -> ModelBackend:
    """Create a backend instance from configuration."""

    resolved_backend = backend_name or config.DEFAULT_BACKEND
    if resolved_backend == "qwen_hf":
        return QwenHFBackend()
    raise ValueError(f"Unsupported backend: {resolved_backend}")


def load_model() -> ModelBackend:
    """Return the configured backend instance."""

    return create_backend().load()


def _coerce_stop_sequences(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value if str(item))


def _truncate_by_stop_sequences(text: str, stop_sequences: Sequence[str]) -> str:
    if not stop_sequences:
        return text.strip()

    earliest_index = None
    for stop_sequence in stop_sequences:
        if not stop_sequence:
            continue
        index = text.find(stop_sequence)
        if index == -1:
            continue
        if earliest_index is None or index < earliest_index:
            earliest_index = index

    if earliest_index is None:
        return text.strip()
    return text[:earliest_index].strip()


def _resolve_runtime_settings(device: str) -> tuple[Any, str | None, str]:
    if device == "cpu" or not torch.cuda.is_available():
        return torch.float32, None, "cpu"

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch_dtype, "auto", "cuda"


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, PromptValue):
        return prompt.to_string()

    if hasattr(prompt, "to_string"):
        return prompt.to_string()

    return str(prompt)


def _prompt_to_messages(prompt: Any) -> list[Any]:
    if hasattr(prompt, "to_messages"):
        return [_normalize_message(message) for message in prompt.to_messages()]

    return [{"role": "user", "content": _prompt_to_text(prompt)}]


def _normalize_message(message: Any) -> dict[str, str]:
    if isinstance(message, Mapping):
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
    else:
        role = str(getattr(message, "type", "user"))
        content = str(getattr(message, "content", message))

    normalized_role = {
        "human": "user",
        "ai": "assistant",
    }.get(role, role)

    return {"role": normalized_role, "content": content}


def _format_prompt_messages(messages: Sequence[Any]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, Mapping):
            role = str(message.get("role", "unknown"))
            content = str(message.get("content", "")).replace("\n", "\\n")
        else:
            role = str(getattr(message, "type", "unknown"))
            content = str(getattr(message, "content", message)).replace("\n", "\\n")
        parts.append(f"{role}: {content}")
    return " | ".join(parts)




def _move_batch_to_device(
    batch: Mapping[str, Any],
    model: Any,
    device_type: str,
) -> dict[str, Any]:
    if device_type != "cpu":
        target_device = getattr(model, "device", None)
        if target_device is None:
            return dict(batch)
        return {
            key: value.to(target_device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    return {
        key: value.to(torch.device("cpu")) if hasattr(value, "to") else value
        for key, value in batch.items()
    }