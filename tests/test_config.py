"""Tests for configuration helpers."""

from pathlib import Path

from src import config


def test_default_backend() -> None:
    assert config.DEFAULT_BACKEND == "qwen_hf"


def test_default_model_name() -> None:
    assert config.DEFAULT_MODEL_NAME == "Qwen/Qwen2.5-1.5B-Instruct"


def test_default_model_path_points_to_local_snapshot() -> None:
    assert Path(config.DEFAULT_MODEL_PATH).exists()