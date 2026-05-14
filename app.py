"""Application entrypoint for the Qwen code generation web app."""

from __future__ import annotations

import logging

import gradio as gr

from src.ui import build_ui


logging.basicConfig(
	level=logging.DEBUG,
	format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


def main() -> None:
	"""Start the application."""

	demo = build_ui()
	demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())


if __name__ == "__main__":
	main()
