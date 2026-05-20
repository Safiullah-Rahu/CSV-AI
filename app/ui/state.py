"""Session-state helpers.

Centralizing access to ``st.session_state`` keeps key names consistent and
makes the chat-history shape easier to refactor later.
"""

from __future__ import annotations

from typing import List

import streamlit as st

from app.llm import Message


def init_state() -> None:
    """Initialize every session key the app reads."""
    defaults = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.2,
        "max_tokens": 1024,
        "api_key": "",
        "ollama_base_url": "http://localhost:11434",
        "dataset": None,        # app.data.loader.LoadedDataset | None
        "profile": None,        # app.data.profiler.DataProfile | None
        "chat_history": [],     # list[Message]
        "analyze_history": [],  # list[Message]
        "summary_text": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_chat() -> None:
    st.session_state["chat_history"] = []


def reset_analyze() -> None:
    st.session_state["analyze_history"] = []


def append_chat(history_key: str, role: str, content: str) -> None:
    history: List[Message] = st.session_state[history_key]
    history.append(Message(role=role, content=content))  # type: ignore[arg-type]
