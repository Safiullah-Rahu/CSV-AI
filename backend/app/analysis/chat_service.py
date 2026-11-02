"""Conversational Q&A over a dataframe.

This replaces the original FAISS + retriever pipeline. For datasets that
fit in the context window we send a structured schema + sample directly;
for larger ones we sample. Modern long-context models make this both
simpler and more accurate than the old embedding-retrieval flow.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import pandas as pd

from app.data import build_dataframe_context
from app.data.profiler import DataProfile, profile_dataframe
from app.llm import LLMProvider, Message
from app.prompts import CHAT_SYSTEM_PROMPT, build_user_question_prompt
from app.utils.logging import get_logger

logger = get_logger("services.chat")


@dataclass
class ChatService:
    """Stateless chat service.

    The conversation history is owned by the caller (the UI keeps it in
    Streamlit session state). That makes the service trivial to reuse from
    a FastAPI route later.
    """

    provider: LLMProvider
    df: pd.DataFrame
    profile: DataProfile | None = None
    context_max_rows: int = 30
    include_correlation: bool = False

    _context: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.profile = self.profile or profile_dataframe(self.df)
        self._context = build_dataframe_context(
            self.df,
            profile=self.profile,
            max_sample_rows=self.context_max_rows,
            include_correlation=self.include_correlation,
        )

    # ------------------------------------------------------------------ API
    def _build_messages(self, history: list[Message], question: str) -> list[Message]:
        system = Message(role="system", content=CHAT_SYSTEM_PROMPT.format(context=self._context))
        user = Message(role="user", content=build_user_question_prompt(question))
        return [system, *history, user]

    def ask(self, question: str, history: list[Message] | None = None) -> str:
        """One-shot answer (non-streaming)."""
        messages = self._build_messages(history or [], question)
        return self.provider.complete(messages).content

    def stream(
        self, question: str, history: list[Message] | None = None
    ) -> Iterator[str]:
        """Token-incremental answer. The caller is responsible for assembly."""
        messages = self._build_messages(history or [], question)
        yield from self.provider.stream(messages)
