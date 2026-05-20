"""Landing page."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("🧠 CSV-AI")
    st.subheader("AI-powered CSV exploration — modernized")

    st.markdown(
        """
        Upload a CSV in the sidebar and pick one of the workflows on the left:

        - **Chat** — ask natural-language questions about your dataset.
        - **Summarize** — generate a concise overview of what the file contains.
        - **Analyze** — inspect schema, missing values, distributions, and
          correlations side-by-side with an LLM analyst.

        This rebuild is **provider-agnostic** (OpenAI, Anthropic, or a local
        Ollama model), uses **modern SDKs**, and separates UI from services
        so the same engine can power a future API.
        """
    )

    a, b, c = st.columns(3)
    with a:
        st.markdown("#### 💬 Chat")
        st.caption("Schema- and sample-aware QA. Streams tokens. Honest about uncertainty.")
    with b:
        st.markdown("#### 📝 Summarize")
        st.caption("Structured overview: contents, key columns, patterns, caveats.")
    with c:
        st.markdown("#### 📊 Analyze")
        st.caption("Deterministic stats + LLM narrative. Charts, correlations, missingness.")

    st.divider()
    if st.session_state.get("dataset") is None:
        st.info("Get started by uploading a CSV in the sidebar.")
    else:
        ds = st.session_state["dataset"]
        st.success(
            f"Dataset `{ds.source_name}` is loaded ({ds.n_rows:,} rows × {ds.n_cols} cols). "
            "Switch to **Chat**, **Summarize**, or **Analyze** from the sidebar."
        )
