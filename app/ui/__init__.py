"""Streamlit UI layer.

The UI is intentionally thin: it owns layout, session state, and rendering
only. All business logic lives in :mod:`app.services`, which means we can
expose the same logic over an API later without rewriting it.
"""
