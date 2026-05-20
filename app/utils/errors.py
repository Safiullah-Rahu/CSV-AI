"""Custom exception hierarchy.

A single base class lets the UI layer catch and render application errors
uniformly without swallowing unrelated bugs.
"""

from __future__ import annotations


class CSVAIError(Exception):
    """Base class for all CSV-AI application errors."""


class ConfigurationError(CSVAIError):
    """Raised when settings or credentials are missing/invalid."""


class ProviderError(CSVAIError):
    """Raised by an LLM provider when a request fails."""


class DataError(CSVAIError):
    """Raised when CSV loading or profiling fails."""
