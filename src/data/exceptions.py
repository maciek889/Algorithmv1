"""Custom exceptions for the data pipeline."""

from __future__ import annotations


class DataLoadError(Exception):
    """Raised when an external data source returns invalid or unusable data."""


class ValidationError(Exception):
    """Raised when assembled data fails an invariant check."""


class ConfigError(Exception):
    """Raised when configuration is missing, malformed, or incomplete."""
