"""Chat page — natural-language QA over the loaded CSV."""

from __future__ import annotations

import streamlit as st

from app.llm import Message
from app.llm.factory import build_provider
from app.services import ChatService
from app.ui.components.chat_box import render_history, stream_assistant
from app.ui.components.data_preview import render_preview
from app.ui.components.sidebar import get_provider_config, require_dataset
from app.ui.state import reset_chat
from app.utils.errors import CSVAIError


def render() -> None:
    st.title("💬 Chat with your CSV")

    dataset = require_dataset()
    if dataset is None:
        return

    render_preview(dataset)
    st.divider()

    col_a, col_b = st.columns([1, 5])
    with col_a:
        if st.button("Reset", use_container_width=True):
            reset_chat()
            st.rerun()
    with col_b:
        st.caption(
            "Answers are grounded in the dataset's schema and a sample of rows. "
            "Streaming is enabled."
        )

    history = st.session_state["chat_history"]
    render_history(history)

    question = st.chat_input("Ask anything about this dataset…")
    if not question:
        return

    provider_name, model, temperature, max_tokens, api_key, base_url = get_provider_config()
    try:
        provider = build_provider(
            provider_name,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )
    except CSVAIError as exc:
        st.error(str(exc))
        return

    service = ChatService(provider=provider, df=dataset.df, profile=st.session_state.get("profile"))

    history.append(Message(role="user", content=question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        text = stream_assistant(lambda: service.stream(question, history[:-1]))

    if text:
        history.append(Message(role="assistant", content=text))
