"""Analytical insights over a DataFrame.

Combines deterministic pandas analysis (always trustworthy) with LLM-
generated narrative (interprets the numbers). This split avoids the most
common failure mode of pure-LLM dataframe agents: confidently wrong
arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import pandas as pd

from app.data import build_dataframe_context
from app.data.profiler import DataProfile, profile_dataframe
from app.llm import LLMProvider, Message
from app.prompts import ANALYZE_SYSTEM_PROMPT, build_user_question_prompt


@dataclass
class AnalysisService:
    """Hybrid analyzer: deterministic stats + LLM narrative."""

    provider: LLMProvider
    df: pd.DataFrame
    profile: Optional[DataProfile] = None

    _context: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.profile = self.profile or profile_dataframe(self.df)
        self._context = build_dataframe_context(
            self.df,
            profile=self.profile,
            max_sample_rows=25,
            include_correlation=True,
        )

    # ----------------------------------------------------- deterministic ops
    def missing_value_report(self) -> pd.DataFrame:
        miss = self.df.isna().sum().to_frame("missing")
        miss["pct"] = (miss["missing"] / len(self.df) * 100).round(2)
        return miss.sort_values("missing", ascending=False)

    def numeric_describe(self) -> pd.DataFrame:
        return self.df.describe(include="number").T

    def categorical_top_values(self, top_n: int = 5) -> Dict[str, pd.Series]:
        cat_cols = self.df.select_dtypes(include=["object", "category", "bool"]).columns
        out: Dict[str, pd.Series] = {}
        for col in cat_cols:
            out[col] = self.df[col].value_counts(dropna=False).head(top_n)
        return out

    def correlation(self) -> Optional[pd.DataFrame]:
        return self.profile.correlation if self.profile else None

    # ----------------------------------------------------- LLM narrative
    def _messages(self, question: str, history: List[Message]) -> List[Message]:
        return [
            Message(role="system", content=ANALYZE_SYSTEM_PROMPT.format(context=self._context)),
            *history,
            Message(role="user", content=build_user_question_prompt(question)),
        ]

    def ask(self, question: str, history: Optional[List[Message]] = None) -> str:
        return self.provider.complete(self._messages(question, history or [])).content

    def stream(
        self, question: str, history: Optional[List[Message]] = None
    ) -> Iterator[str]:
        yield from self.provider.stream(self._messages(question, history or []))
