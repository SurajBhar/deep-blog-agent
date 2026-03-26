"""Default provider construction."""

from __future__ import annotations

from dataclasses import dataclass

from deep_blog_agent.core.settings import AppSettings
from deep_blog_agent.providers.gemini_provider import GeminiImageProvider
from deep_blog_agent.providers.interfaces import ImageProvider, LLMProvider, SearchProvider
from deep_blog_agent.providers.openai_provider import OpenAIChatProvider
from deep_blog_agent.providers.tavily_provider import TavilySearchProvider


@dataclass(frozen=True)
class ProviderBundle:
    """Concrete provider bundle used by the blog workflow."""

    llm: LLMProvider
    search: SearchProvider
    image: ImageProvider


def build_default_provider_bundle(settings: AppSettings) -> ProviderBundle:
    """Build the default provider bundle from app settings."""
    return ProviderBundle(
        llm=OpenAIChatProvider(api_key=settings.openai_api_key, model=settings.openai_model),
        search=TavilySearchProvider(api_key=settings.tavily_api_key),
        image=GeminiImageProvider(api_key=settings.google_api_key, model=settings.google_image_model),
    )
