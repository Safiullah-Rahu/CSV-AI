"""Reusable chat rendering + streaming helper."""

from __future__ import annotations

from typing import Callable, Iterator, List

import streamlit as st

from app.llm import Message


def render_history(history: List[Message]) -> None:
    for msg in history:
        if msg.role == "system":
            continue
        with st.chat_message(msg.role):
            st.markdown(msg.content)


def stream_assistant(
    stream_fn: Callable[[], Iterator[str]],
) -> str:
    """Render an assistant message that updates as tokens stream in.

    Returns the assembled text so the caller can persist it to history.
    """
    placeholder = st.empty()
    buffer: list[str] = []
    try:
        for chunk in stream_fn():
            buffer.append(chunk)
            placeholder.markdown("".join(buffer) + "▌")
        text = "".join(buffer)
        placeholder.markdown(text)
        return text
    except Exception as exc:  # noqa: BLE001
        placeholder.error(f"Model error: {exc}")
        return ""
