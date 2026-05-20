"""Top-level entrypoint for Streamlit Community Cloud.

Streamlit Cloud auto-discovers ``streamlit_app.py``. We keep it tiny — it
just calls into :func:`app.main.run`. All real logic lives in the ``app``
package.
"""

from app.main import run


if __name__ == "__main__":
    run()
else:  # pragma: no cover - Streamlit imports the file directly.
    run()
