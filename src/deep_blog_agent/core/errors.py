"""Application error hierarchy."""


class BlogAgentError(Exception):
    """Base application error."""


class ProviderError(BlogAgentError):
    """Base provider error."""


class ProviderConfigurationError(ProviderError):
    """Raised when a provider is missing required configuration."""


class SearchProviderError(ProviderError):
    """Raised when search execution fails."""


class ImageGenerationError(ProviderError):
    """Raised when image generation fails."""


class ArtifactStoreError(BlogAgentError):
    """Raised when artifact persistence or loading fails."""


class WorkflowExecutionError(BlogAgentError):
    """Raised when workflow execution cannot complete."""
