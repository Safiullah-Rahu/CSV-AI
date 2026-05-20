# Deployment

CSV-AI v2 is structured to deploy three ways out of the box, with a fourth (API split) being a small future task.

## 1. Streamlit Community Cloud

1. Push this repo to GitHub.
2. On [streamlit.io/cloud](https://streamlit.io/cloud), create a new app pointing at this repo.
3. Set **Main file path** to `streamlit_app.py`.
4. Under **Settings → Secrets**, paste:

   ```toml
   OPENAI_API_KEY = "sk-..."
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

5. Deploy. `requirements.txt` is picked up automatically.

> Streamlit Cloud's sandbox can't reach a local Ollama instance — pick `openai` or `anthropic` as your default provider in `Settings → Advanced`.

## 2. Local development

```bash
git clone https://github.com/Safiullah-Rahu/CSV-AI.git
cd CSV-AI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env       # fill in keys
streamlit run streamlit_app.py
```

Optional — run a local Ollama:

```bash
ollama serve &
ollama pull llama3.1
# then in the sidebar: provider = ollama, model = llama3.1
```

## 3. Docker / Docker Compose

```bash
cp .env.example .env       # fill in keys
docker compose up --build
```

Open <http://localhost:8501>. The container runs `streamlit run streamlit_app.py` on port 8501 with a healthcheck.

For a bare-metal Docker run:

```bash
docker build -t csv-ai .
docker run --rm -p 8501:8501 --env-file .env csv-ai
```

## 4. Generic VPS

The Dockerfile is the recommended path on any Linux VPS — drop it behind nginx with an SSL terminator. If you want to skip Docker:

```bash
# As a non-root user on the VPS:
git clone https://github.com/Safiullah-Rahu/CSV-AI.git /opt/csv-ai
cd /opt/csv-ai
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # fill in keys

# Run as a systemd unit (see snippet below):
sudo systemctl enable --now csv-ai.service
```

`csv-ai.service`:

```ini
[Unit]
Description=CSV-AI Streamlit app
After=network.target

[Service]
User=csvai
WorkingDirectory=/opt/csv-ai
EnvironmentFile=/opt/csv-ai/.env
ExecStart=/opt/csv-ai/.venv/bin/streamlit run streamlit_app.py --server.port=8501 --server.address=127.0.0.1
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Front it with nginx:

```nginx
server {
    listen 443 ssl http2;
    server_name csv-ai.example.com;
    # SSL certs ...
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 300;
    }
}
```

## 5. Future: API/backend split

`app/services/` has no Streamlit imports, so wrapping it in FastAPI is mostly mechanical:

```python
# api/main.py — future
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from app.llm.factory import build_provider
from app.services import ChatService

api = FastAPI()

class ChatReq(BaseModel):
    provider: str
    model: str
    question: str
    csv_path: str

@api.post("/chat")
def chat(req: ChatReq):
    provider = build_provider(req.provider, model=req.model)
    df = pd.read_csv(req.csv_path)
    return {"answer": ChatService(provider=provider, df=df).ask(req.question)}
```

Same engine, different transport. The Streamlit app can stay, become a thin client of the API, or both can run side-by-side.

## Health checks & observability

- The Streamlit endpoint `/_stcore/health` returns 200 when ready (used by the Dockerfile healthcheck).
- Logs go to stderr under the `csvai.*` namespace (see `app/utils/logging.py`). Set `LOG_LEVEL=DEBUG` for verbose output.
