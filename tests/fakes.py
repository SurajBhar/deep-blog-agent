from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_blog_agent.blog_writer.contracts import ImageUsageRecord, LLMUsageRecord, SearchResult, SearchUsageRecord
from deep_blog_agent.core.errors import ImageGenerationError
from deep_blog_agent.providers.factory import ProviderBundle
from deep_blog_agent.providers.interfaces import (
    ImageGenerationResult,
    SearchProviderResult,
    StructuredGenerationResult,
    TextGenerationResult,
)


class FakeLLMProvider:
    def __init__(
        self,
        *,
        invoke_responses: list[Any] | None = None,
        structured_responses: list[Any] | None = None,
        invoke_usage: list[Any] | None = None,
        structured_usage: list[Any] | None = None,
    ) -> None:
        self.invoke_responses = list(invoke_responses or [])
        self.structured_responses = list(structured_responses or [])
        self.invoke_usage = list(invoke_usage or [])
        self.structured_usage = list(structured_usage or [])
        self.invocations: list[list[Any]] = []
        self.structured_invocations: list[tuple[list[Any], type]] = []

    def invoke(self, messages):
        self.invocations.append(list(messages))
        response = self.invoke_responses.pop(0)
        if callable(response):
            return response(messages)
        usage = self.invoke_usage.pop(0) if self.invoke_usage else LLMUsageRecord(
            provider="openai",
            step="llm",
            model="gpt-4.1-mini",
            input_tokens=120,
            output_tokens=80,
            total_tokens=200,
        )
        return TextGenerationResult(text=response, usage=usage)

    def invoke_structured(self, messages, schema):
        self.structured_invocations.append((list(messages), schema))
        response = self.structured_responses.pop(0)
        if callable(response):
            response = response(messages, schema)
        if isinstance(response, schema):
            value = response
        elif isinstance(response, dict):
            value = schema(**response)
        else:
            value = response
        usage = self.structured_usage.pop(0) if self.structured_usage else LLMUsageRecord(
            provider="openai",
            step="llm",
            model="gpt-4.1-mini",
            input_tokens=60,
            output_tokens=40,
            total_tokens=100,
        )
        return StructuredGenerationResult(value=value, usage=usage)


class FakeSearchProvider:
    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.responses = responses or {}
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, max_results: int = 5) -> SearchProviderResult:
        self.queries.append((query, max_results))
        response = self.responses.get(query, [])
        if callable(response):
            response = response(query, max_results)
        if isinstance(response, Exception):
            raise response
        return SearchProviderResult(
            results=response,
            usage=SearchUsageRecord(
                provider="tavily",
                step="search",
                model="fake-tavily",
                query=query,
                max_results=max_results,
                result_count=len(response),
                requests=1,
            ),
        )


class FakeImageProvider:
    def __init__(self, responses: dict[str, Any] | None = None, default: bytes | Exception = b"img") -> None:
        self.responses = responses or {}
        self.default = default
        self.prompts: list[str] = []

    def generate_image(self, prompt: str, *, size: str = "1024x1024", quality: str = "medium") -> ImageGenerationResult:
        self.prompts.append(prompt)
        response = self.responses.get(prompt, self.default)
        if callable(response):
            response = response(prompt)
        if isinstance(response, Exception):
            raise response
        if not isinstance(response, bytes):
            raise ImageGenerationError("Fake image provider expected bytes.")
        return ImageGenerationResult(
            image_bytes=response,
            usage=ImageUsageRecord(
                provider="google",
                step="image_generation",
                model="gemini-2.5-flash-image",
                image_count=1,
                size=size,
                quality=quality,
                output_bytes=len(response),
            ),
        )


def make_provider_bundle(
    *,
    llm: FakeLLMProvider | None = None,
    search: FakeSearchProvider | None = None,
    image: FakeImageProvider | None = None,
) -> ProviderBundle:
    return ProviderBundle(
        llm=llm or FakeLLMProvider(invoke_responses=["## Section"], structured_responses=[]),
        search=search or FakeSearchProvider(),
        image=image or FakeImageProvider(),
    )
