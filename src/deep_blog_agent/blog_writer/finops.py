"""Usage aggregation and estimated cost calculation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from deep_blog_agent.blog_writer.contracts import (
    CostLineItem,
    ImageUsageRecord,
    LLMUsageRecord,
    PricingConfig,
    ProviderUsageRecord,
    RunCostSummary,
    SearchUsageRecord,
    UsageRecord,
)


def calculate_cost_summary(
    usage_records: Iterable[UsageRecord | ProviderUsageRecord | dict[str, Any]],
    pricing: PricingConfig,
    *,
    markdown: str = "",
) -> RunCostSummary:
    """Aggregate provider usage into a run-level cost summary."""

    records = [_coerce_usage_record(record) for record in usage_records]
    if not records:
        return RunCostSummary(
            available=False,
            estimated=True,
            currency=pricing.currency,
            pricing=pricing,
            notes=["No usage metadata was captured for this run."],
        )

    by_provider: dict[str, float] = defaultdict(float)
    by_step: dict[str, float] = defaultdict(float)
    line_items: list[CostLineItem] = []
    notes: list[str] = []

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    search_calls = 0
    search_results = 0
    images_generated = 0

    for record in records:
        if isinstance(record, LLMUsageRecord):
            line_item, record_notes = _llm_line_item(record, pricing)
            input_tokens += record.input_tokens or 0
            output_tokens += record.output_tokens or 0
            total_tokens += record.total_tokens or (record.input_tokens or 0) + (record.output_tokens or 0)
        elif isinstance(record, SearchUsageRecord):
            line_item, record_notes = _search_line_item(record, pricing)
            search_calls += record.requests
            search_results += record.result_count
        elif isinstance(record, ImageUsageRecord):
            line_item, record_notes = _image_line_item(record, pricing)
            images_generated += record.image_count
        else:  # pragma: no cover - defensive fallback
            continue

        notes.extend(record_notes)
        line_items.append(line_item)
        by_provider[line_item.provider] += line_item.amount_usd
        by_step[line_item.step] += line_item.amount_usd

    total_estimated_cost_usd = sum(item.amount_usd for item in line_items)
    highest_cost_step = None
    if by_step:
        highest_cost_step = max(by_step.items(), key=lambda item: item[1])[0]

    word_count = len(markdown.split())
    cost_per_1000_words = None
    if word_count > 0:
        cost_per_1000_words = (total_estimated_cost_usd / word_count) * 1000

    return RunCostSummary(
        available=True,
        estimated=True,
        currency=pricing.currency,
        total_estimated_cost_usd=round(total_estimated_cost_usd, 6),
        cost_per_1000_words_usd=round(cost_per_1000_words, 6) if cost_per_1000_words is not None else None,
        by_provider={key: round(value, 6) for key, value in by_provider.items()},
        by_step={key: round(value, 6) for key, value in by_step.items()},
        line_items=line_items,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        search_calls=search_calls,
        search_results=search_results,
        images_generated=images_generated,
        highest_cost_step=highest_cost_step,
        pricing=pricing,
        notes=_unique_notes(notes),
    )


def _coerce_usage_record(value: UsageRecord | ProviderUsageRecord | dict[str, Any]) -> UsageRecord:
    if isinstance(value, (LLMUsageRecord, SearchUsageRecord, ImageUsageRecord)):
        return value
    if isinstance(value, dict):
        usage_type = value.get("usage_type")
        if usage_type == "llm":
            return LLMUsageRecord.model_validate(value)
        if usage_type == "search":
            return SearchUsageRecord.model_validate(value)
        if usage_type == "image":
            return ImageUsageRecord.model_validate(value)
    raise TypeError(f"Unsupported usage record: {value!r}")


def _llm_line_item(record: LLMUsageRecord, pricing: PricingConfig) -> tuple[CostLineItem, list[str]]:
    price = pricing.openai_models.get(record.model or "", None)
    notes: list[str] = []
    amount = 0.0
    if price is None:
        notes.append(f"No LLM price configured for model '{record.model or 'unknown'}'.")
    elif record.input_tokens is None or record.output_tokens is None:
        notes.append(f"LLM token usage missing for step '{record.step}'; LLM cost is partial.")
    else:
        amount = (
            (record.input_tokens / 1_000_000) * price.input_per_1m_tokens_usd
            + (record.output_tokens / 1_000_000) * price.output_per_1m_tokens_usd
        )
    return (
        CostLineItem(
            provider="openai",
            usage_type="llm",
            step=record.step,
            model=record.model,
            description=f"LLM usage for {record.step}",
            amount_usd=round(amount, 6),
        ),
        notes,
    )


def _search_line_item(record: SearchUsageRecord, pricing: PricingConfig) -> tuple[CostLineItem, list[str]]:
    amount = record.requests * pricing.tavily_search.per_query_usd
    return (
        CostLineItem(
            provider="tavily",
            usage_type="search",
            step=record.step,
            model=record.model,
            description=f"Search query: {record.query}",
            amount_usd=round(amount, 6),
        ),
        [],
    )


def _image_line_item(record: ImageUsageRecord, pricing: PricingConfig) -> tuple[CostLineItem, list[str]]:
    price = pricing.google_image_models.get(record.model or "", None)
    amount = 0.0
    notes: list[str] = []
    if price is None:
        notes.append(f"No image price configured for model '{record.model or 'unknown'}'.")
    else:
        amount = record.image_count * price.per_image_usd
    description = f"Image generation for {record.asset_name or record.step}"
    return (
        CostLineItem(
            provider="google",
            usage_type="image",
            step=record.step,
            model=record.model,
            description=description,
            amount_usd=round(amount, 6),
        ),
        notes,
    )


def _unique_notes(notes: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for note in notes:
        if note in seen:
            continue
        seen.add(note)
        output.append(note)
    return output
