"""Statistical panels — missing values, describe, correlations, charts."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data.profiler import DataProfile


def render_stats_tabs(df: pd.DataFrame, profile: DataProfile) -> None:
    """Render a tabbed view: schema, missing, numeric, correlation, charts."""

    tabs = st.tabs(
        ["🧾 Schema", "🕳️ Missing", "📊 Numeric", "🔗 Correlation", "📈 Charts"]
    )

    with tabs[0]:
        _schema_view(profile)

    with tabs[1]:
        _missing_view(df)

    with tabs[2]:
        _numeric_view(df)

    with tabs[3]:
        _correlation_view(profile)

    with tabs[4]:
        _charts_view(df)


def _schema_view(profile: DataProfile) -> None:
    rows = []
    for col in profile.columns:
        rows.append(
            {
                "column": col.name,
                "dtype": col.dtype,
                "missing": col.n_missing,
                "missing %": round(col.pct_missing, 2),
                "unique": col.n_unique,
                "sample": ", ".join(col.sample_values[:3]),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=420)


def _missing_view(df: pd.DataFrame) -> None:
    miss = df.isna().sum().to_frame("missing")
    miss["pct"] = (miss["missing"] / max(len(df), 1) * 100).round(2)
    miss = miss.sort_values("missing", ascending=False)
    st.dataframe(miss, use_container_width=True)
    if miss["missing"].sum() == 0:
        st.success("No missing values detected.")
    else:
        st.bar_chart(miss["pct"])


def _numeric_view(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        st.info("No numeric columns in this dataset.")
        return
    st.dataframe(numeric.describe().T, use_container_width=True)


def _correlation_view(profile: DataProfile) -> None:
    if profile.correlation is None or profile.correlation.empty:
        st.info("Need at least two numeric columns to compute correlations.")
        return
    st.dataframe(profile.correlation.round(3), use_container_width=True)
    try:
        st.write("Heatmap (Pearson):")
        st.dataframe(
            profile.correlation.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
            use_container_width=True,
        )
    except Exception:  # pragma: no cover — styling depends on optional deps
        pass


def _charts_view(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns to chart.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        col = st.selectbox("Histogram column", numeric_cols, key="chart_hist")
        if col:
            st.bar_chart(df[col].dropna().value_counts(bins=20).sort_index())

    with col_b:
        if len(numeric_cols) >= 2:
            x = st.selectbox("X axis", numeric_cols, key="chart_x")
            y = st.selectbox(
                "Y axis",
                [c for c in numeric_cols if c != x] or numeric_cols,
                key="chart_y",
            )
            if x and y and x != y:
                st.scatter_chart(df[[x, y]].dropna(), x=x, y=y)
        else:
            st.caption("Need at least two numeric columns for a scatter plot.")
