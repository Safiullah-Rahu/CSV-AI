"""Exception hierarchy shared by services and the API layer.

Every :class:`CSVAIError` subclass maps to a stable error ``code`` and HTTP
status so API responses are uniform: ``{"error": {"code", "message", ...}}``.
The original CSV-AI names (``CSVAIError``, ``ConfigurationError``,
``ProviderError``, ``DataError``) are kept for compatibility.
"""

from __future__ import annotations

from typing import Any


class CSVAIError(Exception):
    """Base class for all application errors."""

    code = "internal_error"
    status_code = 500

    def __init__(self, message: str = "", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message or self.__class__.__name__)
        self.message = message or self.__class__.__name__
        self.details = details or {}


AppError = CSVAIError


class ConfigurationError(CSVAIError):
    code = "configuration_error"
    status_code = 400


class ProviderError(CSVAIError):
    """An upstream AI provider failed."""

    code = "provider_error"
    status_code = 502


class DataError(CSVAIError):
    """Loading, parsing or profiling data failed."""

    code = "data_error"
    status_code = 422


class ValidationFailed(CSVAIError):
    code = "validation_error"
    status_code = 422


class NotFound(CSVAIError):
    code = "not_found"
    status_code = 404


class Conflict(CSVAIError):
    code = "conflict"
    status_code = 409


class AuthenticationRequired(CSVAIError):
    code = "authentication_required"
    status_code = 401


class PermissionDenied(CSVAIError):
    code = "permission_denied"
    status_code = 403


class RateLimited(CSVAIError):
    code = "rate_limited"
    status_code = 429


class LimitExceeded(CSVAIError):
    """A subscription / usage limit would be exceeded."""

    code = "limit_exceeded"
    status_code = 402


class UnsafeRequest(CSVAIError):
    """Rejected by a safety control (SSRF guard, SQL validator, upload scanner)."""

    code = "unsafe_request"
    status_code = 400


class ConnectorError(CSVAIError):
    code = "connector_error"
    status_code = 502


class NotAvailable(CSVAIError):
    """A capability is not configured or not implemented in this deployment."""

    code = "not_available"
    status_code = 501
