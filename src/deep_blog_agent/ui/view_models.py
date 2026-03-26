"""Pure UI view-model helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import re
from typing import Any, Iterable

from deep_blog_agent.blog_writer.contracts import BlogRunResult, ProviderStatus, RunEvent, SavedBlog

_WORD_RE = re.compile(r"\b[\w'-]+\b")


def format_usd(value: float | None) -> str:
    if value is None:
        return "Cost unavailable"
    return f"${value:,.4f}"


def format_int(value: int | None) -> str:
    if value is None:
        return "0"
    return f"{value:,}"


def format_timestamp(value: str | None) -> str:
    if not value:
        return "Unknown"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if parsed.tzinfo is not None:
        return parsed.strftime("%Y-%m-%d %H:%M UTC")
    return parsed.strftime("%Y-%m-%d %H:%M")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "Runtime unavailable"
    total_seconds = max(int(seconds), 0)
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {remaining_minutes}m"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def markdown_word_count(markdown: str) -> int:
    return len(_WORD_RE.findall(markdown or ""))


def history_rows(saved_blogs: Iterable[SavedBlog]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for saved_blog in saved_blogs:
        cost = saved_blog.cost_summary.total_estimated_cost_usd if saved_blog.cost_summary else None
        rows.append(
            {
                "run_id": saved_blog.run_id,
                "title": saved_blog.title,
                "created_at": saved_blog.created_at or "",
                "created_at_label": format_timestamp(saved_blog.created_at),
                "status": saved_blog.status,
                "source": saved_blog.source,
                "topic": saved_blog.request_topic or saved_blog.title,
                "providers": ", ".join(saved_blog.provider_mix) if saved_blog.provider_mix else "n/a",
                "estimated_cost_usd": cost,
                "estimated_cost_label": format_usd(cost),
            }
        )
    return rows


def finops_rows(saved_blogs: Iterable[SavedBlog]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for saved_blog in saved_blogs:
        cost_summary = saved_blog.cost_summary
        rows.append(
            {
                "run_id": saved_blog.run_id,
                "title": saved_blog.title,
                "created_at": saved_blog.created_at or "",
                "created_at_label": format_timestamp(saved_blog.created_at),
                "estimated_cost_usd": cost_summary.total_estimated_cost_usd if cost_summary else None,
                "cost_per_1000_words_usd": cost_summary.cost_per_1000_words_usd if cost_summary else None,
                "input_tokens": cost_summary.input_tokens if cost_summary else 0,
                "output_tokens": cost_summary.output_tokens if cost_summary else 0,
                "total_tokens": cost_summary.total_tokens if cost_summary else 0,
                "search_calls": cost_summary.search_calls if cost_summary else 0,
                "images_generated": cost_summary.images_generated if cost_summary else 0,
                "highest_cost_step": cost_summary.highest_cost_step if cost_summary else None,
            }
        )
    return rows


def aggregate_provider_costs(saved_blogs: Iterable[SavedBlog]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for saved_blog in saved_blogs:
        if not saved_blog.cost_summary:
            continue
        for provider, amount in saved_blog.cost_summary.by_provider.items():
            totals[provider] += amount
    return dict(totals)


def provider_summary(statuses: Iterable[ProviderStatus]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for status in statuses:
        rows.append(
            {
                "provider": status.provider,
                "state": status.state,
                "ready": status.ready,
                "credential_source": status.credential_source,
                "message": status.message,
                "model": status.model,
            }
        )
    return rows


def summarize_saved_blogs(saved_blogs: Iterable[SavedBlog]) -> dict[str, Any]:
    runs = list(saved_blogs)
    provider_counter: Counter[str] = Counter()
    available_costs: list[float] = []
    total_tokens = 0
    total_search_calls = 0
    total_images = 0

    for saved_blog in runs:
        provider_counter.update(saved_blog.provider_mix)
        if not saved_blog.cost_summary:
            continue
        available_costs.append(saved_blog.cost_summary.total_estimated_cost_usd)
        total_tokens += saved_blog.cost_summary.total_tokens
        total_search_calls += saved_blog.cost_summary.search_calls
        total_images += saved_blog.cost_summary.images_generated

    total_cost = sum(available_costs)
    return {
        "total_runs": len(runs),
        "complete_runs": sum(1 for run in runs if run.status == "complete"),
        "legacy_runs": sum(1 for run in runs if run.source == "legacy"),
        "runs_with_cost": len(available_costs),
        "total_cost": total_cost,
        "average_cost": (total_cost / len(available_costs)) if available_costs else None,
        "total_tokens": total_tokens,
        "total_search_calls": total_search_calls,
        "total_images": total_images,
        "top_provider": provider_counter.most_common(1)[0][0] if provider_counter else "n/a",
    }


def summarize_result(result: BlogRunResult) -> dict[str, Any]:
    cost_summary = result.cost_summary
    tasks = len(result.plan.tasks) if result.plan else 0
    return {
        "word_count": markdown_word_count(result.final_markdown),
        "task_count": tasks,
        "evidence_count": len(result.evidence),
        "warnings_count": len(result.warnings or []),
        "estimated_cost": cost_summary.total_estimated_cost_usd if cost_summary else None,
        "total_tokens": cost_summary.total_tokens if cost_summary else 0,
        "search_calls": cost_summary.search_calls if cost_summary else 0,
        "images_generated": cost_summary.images_generated if cost_summary else 0,
        "highest_cost_step": cost_summary.highest_cost_step if cost_summary else None,
    }


def extract_queries_from_events(events: Iterable[RunEvent]) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for event in events:
        if event.kind != "progress":
            continue
        for query in event.payload.get("queries", []) or []:
            normalized = str(query).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                queries.append(normalized)
    return queries
