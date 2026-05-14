"""Conversation history management."""


def trim_history(history: list[dict], limit: int) -> list[dict]:
    """Keep only the most recent history entries."""

    return history[-limit:]