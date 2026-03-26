"""Tavily-backed search provider."""

from __future__ import annotations

from deep_blog_agent.blog_writer.contracts import SearchResult, SearchUsageRecord
from deep_blog_agent.core.errors import ProviderConfigurationError, SearchProviderError
from deep_blog_agent.providers.interfaces import SearchProviderResult


class TavilySearchProvider:
    """Search provider implemented with tavily-python."""

    def __init__(self, *, api_key: str | None) -> None:
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> SearchProviderResult:
        if not self.api_key:
            raise ProviderConfigurationError("TAVILY_API_KEY is not set.")

        try:
            from tavily import TavilyClient
        except Exception as exc:  # pragma: no cover - import guard
            raise SearchProviderError(f"Unable to import Tavily client: {exc}") from exc

        try:
            client = TavilyClient(api_key=self.api_key)
            response = client.search(query=query, max_results=max_results)
        except Exception as exc:
            raise SearchProviderError(str(exc)) from exc

        results = []
        for item in response.get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title") or "",
                    url=item.get("url") or "",
                    snippet=item.get("content") or item.get("snippet") or "",
                    published_at=item.get("published_date") or item.get("published_at"),
                    source=item.get("source"),
                )
            )
        return SearchProviderResult(
            results=results,
            usage=SearchUsageRecord(
                provider="tavily",
                step="search",
                model="tavily-search",
                query=query,
                max_results=max_results,
                result_count=len(results),
                requests=1,
            ),
        )
