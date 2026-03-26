"""Provider contracts used by the workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, Sequence, TypeVar

from pydantic import BaseModel

from deep_blog_agent.blog_writer.contracts import (
    ImageUsageRecord,
    LLMUsageRecord,
    PromptMessage,
    SearchResult,
    SearchUsageRecord,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class TextGenerationResult:
    text: str
    usage: LLMUsageRecord | None = None


@dataclass(frozen=True)
class StructuredGenerationResult(Generic[ModelT]):
    value: ModelT
    usage: LLMUsageRecord | None = None


@dataclass(frozen=True)
class SearchProviderResult:
    results: list[SearchResult]
    usage: SearchUsageRecord | None = None


@dataclass(frozen=True)
class ImageGenerationResult:
    image_bytes: bytes
    usage: ImageUsageRecord | None = None


class LLMProvider(Protocol):
    """Structured and unstructured LLM access."""

    def invoke(self, messages: Sequence[PromptMessage]) -> TextGenerationResult:
        """Run a plain text generation call."""

    def invoke_structured(self, messages: Sequence[PromptMessage], schema: type[ModelT]) -> StructuredGenerationResult[ModelT]:
        """Run a structured generation call."""


class SearchProvider(Protocol):
    """Search interface for research retrieval."""

    def search(self, query: str, max_results: int = 5) -> SearchProviderResult:
        """Search and return normalized results."""


class ImageProvider(Protocol):
    """Image generation interface."""

    def generate_image(self, prompt: str, *, size: str = "1024x1024", quality: str = "medium") -> ImageGenerationResult:
        """Generate raw image bytes."""
