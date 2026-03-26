"""OpenAI-backed LLM provider."""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel

from deep_blog_agent.blog_writer.contracts import LLMUsageRecord, PromptMessage
from deep_blog_agent.core.errors import ProviderConfigurationError, ProviderError
from deep_blog_agent.providers.interfaces import (
    ModelT,
    StructuredGenerationResult,
    TextGenerationResult,
)


class OpenAIChatProvider:
    """LLM provider implemented with langchain-openai."""

    def __init__(self, *, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY is not set.")
        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:  # pragma: no cover - import guard
            raise ProviderError(f"Unable to import langchain_openai: {exc}") from exc
        self._client = ChatOpenAI(model=self.model, api_key=self.api_key)
        return self._client

    @staticmethod
    def _to_langchain_messages(messages: Sequence[PromptMessage]):
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except Exception as exc:  # pragma: no cover - import guard
            raise ProviderError(f"Unable to import langchain_core messages: {exc}") from exc

        output = []
        for message in messages:
            if message.role == "system":
                output.append(SystemMessage(content=message.content))
            else:
                output.append(HumanMessage(content=message.content))
        return output

    @staticmethod
    def _extract_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            return "\n".join(parts).strip()
        return str(content)

    def invoke(self, messages: Sequence[PromptMessage]) -> TextGenerationResult:
        client = self._get_client()
        response = client.invoke(self._to_langchain_messages(messages))
        return TextGenerationResult(
            text=self._extract_content(response.content).strip(),
            usage=self._extract_usage(response),
        )

    def invoke_structured(self, messages: Sequence[PromptMessage], schema: type[ModelT]) -> StructuredGenerationResult[ModelT]:
        base_client = self._get_client()
        try:
            client = base_client.with_structured_output(schema, include_raw=True)
        except TypeError:  # pragma: no cover - compatibility fallback
            client = base_client.with_structured_output(schema)
        response = client.invoke(self._to_langchain_messages(messages))
        parsed = response
        raw = response
        if isinstance(response, dict):
            parsed = response.get("parsed")
            raw = response.get("raw")

        if not isinstance(parsed, BaseModel):
            raise ProviderError(f"Structured response did not match schema {schema.__name__}.")
        return StructuredGenerationResult(value=parsed, usage=self._extract_usage(raw))

    def _extract_usage(self, response: object) -> LLMUsageRecord:
        usage_metadata = getattr(response, "usage_metadata", None)
        response_metadata = getattr(response, "response_metadata", None)

        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        if isinstance(usage_metadata, dict):
            prompt_tokens = usage_metadata.get("input_tokens")
            completion_tokens = usage_metadata.get("output_tokens")
            total_tokens = usage_metadata.get("total_tokens")

        if isinstance(response_metadata, dict):
            token_usage = response_metadata.get("token_usage")
            if isinstance(token_usage, dict):
                prompt_tokens = prompt_tokens if prompt_tokens is not None else token_usage.get("prompt_tokens")
                completion_tokens = (
                    completion_tokens if completion_tokens is not None else token_usage.get("completion_tokens")
                )
                total_tokens = total_tokens if total_tokens is not None else token_usage.get("total_tokens")

        estimated = prompt_tokens is None and completion_tokens is None and total_tokens is None
        return LLMUsageRecord(
            provider="openai",
            step="llm",
            model=self.model,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated=estimated,
            metadata={"usage_available": not estimated},
        )
