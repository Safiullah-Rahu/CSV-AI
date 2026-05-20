"""Generate a human-readable summary of a CSV.

Original CSV-AI used LangChain's ``load_summarize_chain(map_reduce)`` over
the CSV text — slow, expensive, and shape-blind. We instead build a
schema-plus-stats prompt and let a single model call produce the summary.
For very wide datasets we sample columns to keep token usage bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import pandas as pd

from app.data import build_dataframe_context
from app.data.profiler import DataProfile, profile_dataframe
from app.llm import LLMProvider, Message
from app.prompts import SUMMARY_SYSTEM_PROMPT


@dataclass
class SummaryService:
    provider: LLMProvider
    df: pd.DataFrame
    profile: Optional[DataProfile] = None

    def __post_init__(self) -> None:
        self.profile = self.profile or profile_dataframe(self.df)

    def _messages(self) -> list[Message]:
        context = build_dataframe_context(
            self.df,
            profile=self.profile,
            max_sample_rows=20,
            include_correlation=False,
        )
        return [
            Message(role="system", content=SUMMARY_SYSTEM_PROMPT.format(context=context)),
            Message(role="user", content="Please produce the dataset summary."),
        ]

    def generate(self) -> str:
        return self.provider.complete(self._messages()).content

    def stream(self) -> Iterator[str]:
        yield from self.provider.stream(self._messages())
