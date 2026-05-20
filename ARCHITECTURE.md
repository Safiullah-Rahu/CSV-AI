# Architecture

CSV-AI v2 is a layered Streamlit application designed so that the **business logic is independent of the UI**. The same `services/` package could be served from FastAPI, a CLI, or a Discord bot tomorrow without changes.

```
┌───────────────────────────────────────────────────────────────────┐
│                         streamlit_app.py                          │
│                          (thin entry)                             │
└──────────────────────────┬────────────────────────────────────────┘
                           │
                  ┌────────▼────────┐
                  │     app.ui      │
                  │  pages + comps  │   ← Streamlit lives here only
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  app.services   │
                  │  Chat·Summary·  │   ← pure Python; UI-free
                  │     Analysis    │
                  └────────┬────────┘
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │  app.data  │  │  app.llm   │  │ app.prompts │
    │ load·prof  │  │  provider  │  │  templates  │
    │  sample    │  │   layer    │  │             │
    └────────────┘  └────┬───────┘  └────────────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
        OpenAI       Anthropic      Ollama
```

## Why each layer exists

### `app.config`
Single source of truth for settings via `pydantic-settings`. Reads `.env`, OS env, and (best-effort) Streamlit secrets. Anything that needs an API key or a default should call `get_settings()` rather than reading `os.environ`.

### `app.llm`
A provider-agnostic interface (`LLMProvider`) with three concrete implementations. Adding a new backend (Mistral API, Bedrock, …) is one file plus one factory entry. Keeping streaming in the interface forces every provider to behave consistently for the chat UI.

### `app.data`
Pure-pandas. **Never** imports Streamlit or any LLM SDK. Three concerns:
- `loader.py` — robust CSV reading (encoding fallback chain, delimiter sniffing).
- `profiler.py` — fast statistical profile (rows, dtypes, missing %, correlations, samples).
- `sampler.py` + `context.py` — turn a DataFrame into an LLM-friendly markdown context.

### `app.prompts`
System prompts as constants. Versioning prompts in code (not strings sprinkled through services) keeps prompt iteration reviewable in git.

### `app.services`
Three services map 1:1 to the three product workflows. Each takes a `provider` and a `df`, exposes a non-streaming and streaming method, and never imports Streamlit. They're trivially callable from FastAPI later.

### `app.ui`
The only place we import Streamlit. Page modules render; component modules are reusable. Session state is centralized in `state.py` so key names stay consistent.

### `app.utils`
Cross-cutting: logging, exception hierarchy, token counting. No business logic.

## Key design choices

**No FAISS / no retrieval for chat.** The original CSV-AI embedded every row chunk into FAISS and retrieved on every question. For small files this was wasted compute; for analytical questions ("what's the average X?") it was actively worse than just showing the model the schema. v2 sends a structured schema + a smart sample. Result: lower latency, lower cost, fewer hallucinations.

**Deterministic stats first, LLM second.** The Analyze page computes the actual numbers with pandas, then asks the LLM to write the narrative around those numbers. The LLM never invents arithmetic — pure-LLM dataframe agents do.

**Streaming everywhere.** Every provider yields token chunks. The UI uses `st.empty()` plus `st.markdown` with a trailing cursor for the classic typewriter feel.

**Custom-model escape hatch.** The sidebar's model picker has a `custom...` option so new model ids (e.g. tomorrow's `gpt-5`) work without a code change.

**Future API split.** Because services don't import Streamlit, exposing an HTTP layer is mostly:

```python
@app.post("/chat")
def chat(req: ChatReq) -> ChatResp:
    provider = build_provider(req.provider, model=req.model, ...)
    df = pd.read_csv(req.csv_path)
    service = ChatService(provider=provider, df=df)
    return ChatResp(answer=service.ask(req.question))
```

## Adding a new LLM provider

1. Create `app/llm/<name>_provider.py` subclassing `LLMProvider`. Implement `complete` and `stream`.
2. Register it in `app/llm/factory.py` (`_PROVIDERS`).
3. Add suggested models to `_MODEL_CATALOGUE` in the same file.
4. Extend `Settings` with any new credential fields.
5. Update the sidebar's `_render_provider_section` if it needs a non-standard field (e.g. a region).

## Adding a new page

1. Create `app/ui/pages/<name>.py` with a `render()` function.
2. Add it to `PAGE_RENDERERS` in `app/main.py` and `PAGES` in `app/ui/components/sidebar.py`.

That's it. No central router to wire up.
