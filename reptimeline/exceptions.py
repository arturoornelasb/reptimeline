"""Domain-specific exceptions for reptimeline."""


class ReptimelineError(Exception):
    """Base exception for all reptimeline errors."""


class SnapshotError(ReptimelineError):
    """Invalid or empty snapshot data."""


class ExtractionError(ReptimelineError):
    """Error during representation extraction."""


class DiscoveryError(ReptimelineError):
    """Error during bit discovery or analysis."""


class ConfigurationError(ReptimelineError):
    """Invalid configuration or parameter values."""
