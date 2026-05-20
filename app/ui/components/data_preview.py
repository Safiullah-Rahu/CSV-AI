"""Dataset preview card (top-of-page on Chat / Summarize / Analyze)."""

from __future__ import annotations

import streamlit as st

from app.data.loader import LoadedDataset


def render_preview(dataset: LoadedDataset, *, expanded: bool = False) -> None:
    """Render an expandable preview with key metrics + the head of the frame."""

    cols = st.columns(4)
    cols[0].metric("Rows", f"{dataset.n_rows:,}")
    cols[1].metric("Columns", dataset.n_cols)
    cols[2].metric("Encoding", dataset.encoding)
    cols[3].metric("Delimiter", repr(dataset.delimiter))

    with st.expander("📄 Preview rows", expanded=expanded):
        st.dataframe(dataset.df.head(50), use_container_width=True, height=320)
