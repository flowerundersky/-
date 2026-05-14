"""Configuration helpers for the application."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_BACKEND = os.getenv("BACKEND", "qwen_hf")

DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
DEFAULT_LOCAL_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-3B" / "models--Qwen--Qwen2.5-3B-Instruct-AWQ" / "snapshots" / "3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd"
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_LOCAL_MODEL_PATH))

DEFAULT_DEVICE = os.getenv("DEVICE", "auto")

DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
DEFAULT_MIN_NEW_TOKENS = int(os.getenv("MIN_NEW_TOKENS", "0"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.getenv("TOP_P", "0.90"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.05"))