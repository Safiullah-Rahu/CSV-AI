# CSV-AI 🧠 v2

**Modernized AI-powered CSV analysis** — chat with, summarize, and analyze your CSV files using OpenAI, Anthropic, or a local Ollama model. Built for Streamlit Cloud, local laptops, and future API split.

> This is the v2 rewrite of [Safiullah-Rahu/CSV-AI](https://github.com/Safiullah-Rahu/CSV-AI). The product idea is unchanged; the architecture is modular, the AI stack is provider-agnostic, and the UI is a clean modern dashboard.

## Features

- **💬 Chat** — schema- and sample-aware Q&A with token-streaming.
- **📝 Summarize** — single-call structured overview (replaces the old map-reduce flow).
- **📊 Analyze** — deterministic pandas stats + LLM analyst narrative side-by-side, with charts, missingness, and correlations.
- **🔌 Multi-provider** — OpenAI, Anthropic Claude, or local Ollama.
- **🎛️ Modern UI** — sidebar nav, `st.chat_message`, light/dark friendly, custom CSS polish.
- **🧱 Modular** — clean `app/` package; no Streamlit imports in services, so a FastAPI layer can be added later.

## Quick start

```bash
git clone https://github.com/Safiullah-Rahu/CSV-AI.git
cd CSV-AI

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env       # then add your API keys
streamlit run streamlit_app.py
```

Open <http://localhost:8501> and upload a CSV.

## Configuration

CSV-AI loads settings from (in order): environment variables → `.env` file → Streamlit secrets. See `.env.example` and `.streamlit/secrets.toml.example`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Required if using OpenAI. |
| `ANTHROPIC_API_KEY` | — | Required if using Anthropic. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama endpoint. |
| `DEFAULT_PROVIDER` | `openai` | One of `openai`, `anthropic`, `ollama`. |
| `DEFAULT_MODEL` | `gpt-4o-mini` | Used until the user picks one in the sidebar. |
| `DEFAULT_TEMPERATURE` | `0.2` | 0.0–1.5. |
| `DEFAULT_MAX_TOKENS` | `1024` | Response length cap. |

## Project structure

```
app/
├── config/        # pydantic-settings (env + secrets)
├── llm/           # provider-agnostic LLM interface + OpenAI / Anthropic / Ollama
├── data/          # CSV loader, profiler, sampler, prompt-context builder
├── prompts/       # versioned system prompts
├── services/      # ChatService, SummaryService, AnalysisService (UI-free)
├── ui/            # Streamlit pages + components + theme + session state
└── utils/         # logging, errors, token counting

tests/             # pytest suite (loader, profiler, factory)
streamlit_app.py   # Streamlit Cloud entry point
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the rationale behind each layer.

## Deployment

- **Streamlit Community Cloud** — point it at `streamlit_app.py`, add keys to `Secrets`.
- **Local** — `streamlit run streamlit_app.py`.
- **Docker** — `docker compose up --build` (uses `.env`).
- **Future API split** — services are pure-Python; a FastAPI layer is a small adapter.

Full instructions in [DEPLOYMENT.md](DEPLOYMENT.md).

## Development

```bash
pip install -r requirements-dev.txt
pytest                     # run tests
ruff check .               # lint
black .                    # format
```

## What changed vs. v1

| | v1 | v2 |
| --- | --- | --- |
| Architecture | 278-line `app.py` | modular `app/` package |
| LLM | OpenAI only, via LangChain | OpenAI · Anthropic · Ollama via thin native SDKs |
| Imports | `langchain.chat_models`, `langchain.embeddings` (deprecated) | current SDKs |
| Chat context | FAISS retrieval over CSV chunks | schema + smart sample (cheaper, more accurate) |
| Summarize | LangChain `load_summarize_chain(map_reduce)` | single structured prompt |
| Analyze | `create_pandas_dataframe_agent` only | deterministic pandas stats + LLM narrative |
| UI | one `selectbox` of "functionality" | sidebar nav + tabbed stats + theme polish |
| Config | `os.environ` inline | `pydantic-settings` |
| Tests | none | pytest suite |
| Docker | none | Dockerfile + compose |

## License

MIT — see [LICENSE](LICENSE).
