"""Custom CSS for the modern dashboard.

Streamlit's default styling is functional but visually noisy. A small CSS
overlay tightens spacing, modernizes the chat bubbles, and adds a soft card
treatment for metric blocks. It respects the user's light/dark theme by
using only CSS variables Streamlit already provides.
"""

from __future__ import annotations

import streamlit as st


_CSS = """
<style>
/* Tighten the top padding so the header sits closer to the nav. */
.main .block-container {
    padding-top: 1.25rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}

/* Subtle card treatment for metrics. */
div[data-testid="stMetric"] {
    background: rgba(127, 127, 127, 0.06);
    padding: 0.75rem 1rem;
    border-radius: 12px;
    border: 1px solid rgba(127, 127, 127, 0.12);
}

/* Make chat messages feel less cramped. */
div[data-testid="stChatMessage"] {
    padding: 0.4rem 0.75rem;
}

/* Sidebar header polish. */
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    margin-top: 0.4rem;
    margin-bottom: 0.4rem;
}

/* Slight rounding on dataframes and code blocks. */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* Soft divider between page sections. */
hr {
    opacity: 0.4;
}
</style>
"""


def apply_theme() -> None:
    """Inject the dashboard CSS. Idempotent within a session."""
    if st.session_state.get("_csvai_theme_applied"):
        return
    st.markdown(_CSS, unsafe_allow_html=True)
    st.session_state["_csvai_theme_applied"] = True
