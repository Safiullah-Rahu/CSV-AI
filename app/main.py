"""Streamlit application entry point.

Run with::

    streamlit run streamlit_app.py

Or, equivalently::

    streamlit run -m app.main

This module is intentionally tiny — all rendering lives in ``app.ui.pages``
so each page is independently testable.
"""

from __future__ import annotations

import streamlit as st

from app.ui.components.sidebar import render_sidebar
from app.ui.pages import analyze, chat, home, summarize
from app.ui.state import init_state
from app.ui.theme import apply_theme


PAGE_RENDERERS = {
    "Home": home.render,
    "Chat": chat.render,
    "Summarize": summarize.render,
    "Analyze": analyze.render,
}


def run() -> None:
    st.set_page_config(
        page_title="CSV-AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    init_state()

    page = render_sidebar()
    renderer = PAGE_RENDERERS.get(page, home.render)
    renderer()


if __name__ == "__main__":
    run()
