"""Sidebar: navigation + provider/model configuration + file upload.

The sidebar is the only place the user enters credentials or switches
provider. Keeping that consolidated avoids the original app's pattern of
re-uploading the file on every page.
"""

from __future__ import annotations

from typing import Optional, Tuple

import streamlit as st

from app.config import get_settings
from app.data.loader import LoadedDataset, load_csv
from app.data.profiler import profile_dataframe
from app.llm.factory import available_models, available_providers
from app.utils.errors import DataError


PAGES = ["Home", "Chat", "Summarize", "Analyze"]


def render_sidebar() -> str:
    """Render the sidebar and return the currently selected page name."""
    settings = get_settings()

    with st.sidebar:
        st.markdown("## 🧠 CSV-AI")
        st.caption("Modernized • multi-provider • production-ready")
        st.divider()

        page = st.radio("Navigate", PAGES, key="nav_page", label_visibility="collapsed")

        st.divider()
        _render_provider_section(settings)
        st.divider()
        _render_upload_section()
        st.divider()
        st.caption(
            "Code is open-source on "
            "[GitHub](https://github.com/Safiullah-Rahu/CSV-AI). "
            "Modernized v2."
        )

    return page


def _render_provider_section(settings) -> None:
    st.subheader("Model")
    provider = st.selectbox(
        "Provider",
        available_providers(),
        index=available_providers().index(st.session_state.get("provider", settings.default_provider)),
        key="provider",
    )

    suggested_models = available_models(provider)
    current_model = st.session_state.get("model") or settings.default_model
    if current_model not in suggested_models and suggested_models:
        current_model = suggested_models[0]

    model = st.selectbox(
        "Model",
        suggested_models + ["custom..."],
        index=suggested_models.index(current_model) if current_model in suggested_models else 0,
        key="model_select",
    )
    if model == "custom...":
        model = st.text_input(
            "Custom model id",
            value=st.session_state.get("model", current_model),
            key="custom_model",
        )
    st.session_state["model"] = model

    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=float(st.session_state.get("temperature", settings.default_temperature)),
        step=0.05,
        key="temperature",
    )
    st.slider(
        "Max tokens",
        min_value=128,
        max_value=8000,
        value=int(st.session_state.get("max_tokens", settings.default_max_tokens)),
        step=64,
        key="max_tokens",
    )

    if provider == "openai":
        st.text_input(
            "OpenAI API key",
            value=st.session_state.get("api_key", settings.openai_api_key or ""),
            type="password",
            key="api_key",
            help="Loaded from .env / Streamlit secrets if available.",
        )
    elif provider == "anthropic":
        st.text_input(
            "Anthropic API key",
            value=st.session_state.get("api_key", settings.anthropic_api_key or ""),
            type="password",
            key="api_key",
        )
    elif provider == "ollama":
        st.text_input(
            "Ollama base URL",
            value=st.session_state.get("ollama_base_url", settings.ollama_base_url),
            key="ollama_base_url",
            help="Local-only. Start it with `ollama serve` and pull the model first.",
        )


def _render_upload_section() -> None:
    st.subheader("Dataset")
    uploaded = st.file_uploader(
        "Upload a CSV",
        type=["csv", "tsv", "txt"],
        accept_multiple_files=False,
        key="uploaded_csv",
    )
    if uploaded is None:
        if st.session_state.get("dataset") is not None:
            st.caption(f"Loaded: **{st.session_state['dataset'].source_name}**")
        return

    try:
        dataset: LoadedDataset = load_csv(uploaded, source_name=uploaded.name)
    except DataError as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    st.session_state["dataset"] = dataset
    st.session_state["profile"] = profile_dataframe(dataset.df)
    st.success(
        f"Loaded `{dataset.source_name}` — {dataset.n_rows:,} rows × {dataset.n_cols} cols "
        f"(encoding: {dataset.encoding}, sep: {dataset.delimiter!r})"
    )


def require_dataset() -> Optional[LoadedDataset]:
    """Return the loaded dataset or render an info message and ``None``."""
    dataset: Optional[LoadedDataset] = st.session_state.get("dataset")
    if dataset is None:
        st.info("⬅️ Upload a CSV from the sidebar to get started.")
        return None
    return dataset


def get_provider_config() -> Tuple[str, str, float, int, Optional[str], Optional[str]]:
    """Return (provider, model, temperature, max_tokens, api_key, base_url)."""
    provider = st.session_state["provider"]
    model = st.session_state["model"]
    temperature = float(st.session_state["temperature"])
    max_tokens = int(st.session_state["max_tokens"])
    api_key = st.session_state.get("api_key") or None
    base_url = st.session_state.get("ollama_base_url") if provider == "ollama" else None
    return provider, model, temperature, max_tokens, api_key, base_url
