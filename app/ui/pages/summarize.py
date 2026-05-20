"""Summarize page — single-shot dataset overview."""

from __future__ import annotations

import streamlit as st

from app.llm.factory import build_provider
from app.services import SummaryService
from app.ui.components.chat_box import stream_assistant
from app.ui.components.data_preview import render_preview
from app.ui.components.sidebar import get_provider_config, require_dataset
from app.utils.errors import CSVAIError


def render() -> None:
    st.title("📝 Summarize")

    dataset = require_dataset()
    if dataset is None:
        return

    render_preview(dataset)
    st.divider()

    col_a, col_b = st.columns([1, 5])
    with col_a:
        generate = st.button("Generate", type="primary", use_container_width=True)
    with col_b:
        st.caption(
            "One LLM call over a structured schema + sample. Far cheaper "
            "than the old map-reduce-over-rows approach."
        )

    if generate:
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

        service = SummaryService(provider=provider, df=dataset.df, profile=st.session_state.get("profile"))
        with st.status("Generating summary…", expanded=True) as status:
            text = stream_assistant(service.stream)
            status.update(label="Summary ready.", state="complete")
        st.session_state["summary_text"] = text

    if st.session_state.get("summary_text"):
        st.divider()
        st.markdown("### Latest summary")
        st.markdown(st.session_state["summary_text"])
        st.download_button(
            "Download as markdown",
            data=st.session_state["summary_text"],
            file_name=f"{dataset.source_name.rsplit('.', 1)[0]}_summary.md",
            mime="text/markdown",
        )
