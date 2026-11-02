"""Google Gemini (REST) and AWS Bedrock (Converse API via boto3) providers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

from app.llm.base import ChatResponse, LLMProvider, Message
from app.utils.errors import ConfigurationError, NotAvailable, ProviderError


class GeminiProvider(LLMProvider):
    name = "gemini"

    def _req(self, messages: Iterable[Message]) -> tuple[dict[str, Any], dict[str, str]]:
        if not self.api_key:
            raise ConfigurationError("Gemini API key is missing.")
        system, contents = [], []
        for m in self._normalize(messages):
            if m["role"] == "system":
                system.append(m["content"])
            else:
                contents.append({"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]})
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": self.temperature, "maxOutputTokens": self.max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
        return payload, {"x-goog-api-key": self.api_key}

    def _base(self) -> str:
        return (self.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

    def list_models(self) -> list[str]:
        """``models.list``: chat (generateContent) and embedding (embedContent) models."""
        if not self.api_key:
            raise ConfigurationError("Gemini API key is missing.")
        out: list[str] = []
        token: str | None = None
        for _ in range(10):
            data = self._get_json(f"{self._base()}/models?pageSize=200" + (f"&pageToken={token}" if token else ""), {"x-goog-api-key": self.api_key})
            for m in data.get("models") or []:
                methods = m.get("supportedGenerationMethods") or []
                if "generateContent" in methods or "embedContent" in methods:
                    out.append(str(m.get("name", "")).removeprefix("models/"))
            token = data.get("nextPageToken")
            if not token:
                break
        return out

    def complete(self, messages: Iterable[Message]) -> ChatResponse:
        payload, headers = self._req(messages)
        data = self._post_json(f"{self._base()}/models/{self.model}:generateContent", payload, headers)
        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        usage = data.get("usageMetadata") or {}
        return ChatResponse(
            content="".join(p.get("text", "") for p in parts),
            model=self.model,
            prompt_tokens=int(usage.get("promptTokenCount") or 0),
            completion_tokens=int(usage.get("candidatesTokenCount") or 0),
            raw=data,
        )

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        payload, headers = self._req(messages)
        self.last_usage = None
        for line in self._stream_lines(f"{self._base()}/models/{self.model}:streamGenerateContent?alt=sse", payload, headers):
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            usage = event.get("usageMetadata")
            if usage:
                self.last_usage = (int(usage.get("promptTokenCount") or 0), int(usage.get("candidatesTokenCount") or 0))
            for cand in event.get("candidates") or []:
                for part in (cand.get("content") or {}).get("parts") or []:
                    if part.get("text"):
                        yield part["text"]


class BedrockProvider(LLMProvider):
    """AWS Bedrock Converse API. Needs ``boto3`` and AWS credentials (env, profile or role)."""

    name = "bedrock"

    def _client(self) -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise NotAvailable("AWS Bedrock needs boto3 (pip extra 's3' or install boto3).") from exc
        kwargs: dict[str, Any] = {"region_name": self.options.get("region", "us-east-1")}
        if self.options.get("access_key_id") and self.api_key:
            kwargs.update(aws_access_key_id=self.options["access_key_id"], aws_secret_access_key=self.api_key)
        return boto3.client("bedrock-runtime", **kwargs)

    def _split(self, messages: Iterable[Message]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        system, convo = [], []
        for m in self._normalize(messages):
            if m["role"] == "system":
                system.append({"text": m["content"]})
            else:
                convo.append({"role": m["role"], "content": [{"text": m["content"]}]})
        return system, convo

    def complete(self, messages: Iterable[Message]) -> ChatResponse:
        system, convo = self._split(messages)
        try:
            resp = self._client().converse(
                modelId=self.model, messages=convo, system=system,
                inferenceConfig={"maxTokens": self.max_tokens, "temperature": self.temperature},
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Bedrock request failed: {exc}") from exc
        content = "".join(c.get("text", "") for c in resp["output"]["message"]["content"])
        usage = resp.get("usage", {})
        return ChatResponse(content=content, model=self.model, prompt_tokens=usage.get("inputTokens", 0), completion_tokens=usage.get("outputTokens", 0), raw=resp)

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        system, convo = self._split(messages)
        try:
            resp = self._client().converse_stream(
                modelId=self.model, messages=convo, system=system,
                inferenceConfig={"maxTokens": self.max_tokens, "temperature": self.temperature},
            )
            for event in resp["stream"]:
                delta = event.get("contentBlockDelta", {}).get("delta", {}).get("text")
                if delta:
                    yield delta
                if "metadata" in event:
                    u = event["metadata"].get("usage", {})
                    self.last_usage = (u.get("inputTokens", 0), u.get("outputTokens", 0))
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Bedrock stream failed: {exc}") from exc
