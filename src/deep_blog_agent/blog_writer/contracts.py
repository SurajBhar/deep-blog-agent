"""Workflow contracts and state models."""

from __future__ import annotations

import operator
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class PromptMessage(BaseModel):
    role: Literal["system", "user"]
    content: str


class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do or understand.")
    bullets: list[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120-550).")
    tags: list[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: list[str] = Field(default_factory=list)
    tasks: list[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None
    published_at: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: list[str] = Field(default_factory=list)
    max_results_per_query: int = Field(default=5)


class EvidencePack(BaseModel):
    evidence: list[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: list[ImageSpec] = Field(default_factory=list)


class LLMPriceConfig(BaseModel):
    input_per_1m_tokens_usd: float = 0.0
    output_per_1m_tokens_usd: float = 0.0


class SearchPriceConfig(BaseModel):
    per_query_usd: float = 0.0


class ImagePriceConfig(BaseModel):
    per_image_usd: float = 0.0


class PricingConfig(BaseModel):
    currency: str = "USD"
    label: str = "Application defaults"
    openai_models: dict[str, LLMPriceConfig] = Field(default_factory=dict)
    tavily_search: SearchPriceConfig = Field(default_factory=SearchPriceConfig)
    google_image_models: dict[str, ImagePriceConfig] = Field(default_factory=dict)


class SessionRuntimeConfig(BaseModel):
    openai_api_key: Optional[str] = Field(default=None, exclude=True, repr=False)
    tavily_api_key: Optional[str] = Field(default=None, exclude=True, repr=False)
    google_api_key: Optional[str] = Field(default=None, exclude=True, repr=False)
    langsmith_api_key: Optional[str] = Field(default=None, exclude=True, repr=False)
    openai_model: Optional[str] = None
    google_image_model: Optional[str] = None
    default_enable_research: Optional[bool] = None
    default_enable_images: Optional[bool] = None
    langsmith_tracing: Optional[bool] = None
    langsmith_project: Optional[str] = None
    pricing: Optional[PricingConfig] = None


class ResolvedRuntimeConfig(BaseModel):
    openai_model: str
    google_image_model: str
    default_enable_research: bool = True
    default_enable_images: bool = True
    langsmith_tracing: bool = False
    langsmith_project: Optional[str] = None
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    credential_sources: dict[str, Literal["session", "deployment", "missing"]] = Field(default_factory=dict)


class ProviderStatus(BaseModel):
    provider: Literal["openai", "tavily", "google"]
    state: Literal["using_deployment_default", "using_session_override", "missing", "validation_failed"]
    ready: bool
    credential_source: Literal["session", "deployment", "missing"]
    message: str
    model: Optional[str] = None


class BlogRequest(BaseModel):
    topic: str
    as_of: date
    enable_research: bool = True
    enable_images: bool = True
    runtime_overrides: Optional[SessionRuntimeConfig] = None


class ProviderUsageRecord(BaseModel):
    provider: Literal["openai", "tavily", "google"]
    usage_type: Literal["llm", "search", "image"]
    step: str
    model: Optional[str] = None
    estimated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMUsageRecord(ProviderUsageRecord):
    usage_type: Literal["llm"] = "llm"
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class SearchUsageRecord(ProviderUsageRecord):
    usage_type: Literal["search"] = "search"
    query: str
    max_results: int = 5
    result_count: int = 0
    requests: int = 1


class ImageUsageRecord(ProviderUsageRecord):
    usage_type: Literal["image"] = "image"
    image_count: int = 1
    size: Optional[str] = None
    quality: Optional[str] = None
    output_bytes: Optional[int] = None
    asset_name: Optional[str] = None


UsageRecord = Annotated[LLMUsageRecord | SearchUsageRecord | ImageUsageRecord, Field(discriminator="usage_type")]


class CostLineItem(BaseModel):
    provider: Literal["openai", "tavily", "google"]
    usage_type: Literal["llm", "search", "image"]
    step: str
    model: Optional[str] = None
    description: str
    amount_usd: float = 0.0


class RunCostSummary(BaseModel):
    available: bool = False
    estimated: bool = True
    currency: str = "USD"
    total_estimated_cost_usd: float = 0.0
    cost_per_1000_words_usd: Optional[float] = None
    by_provider: dict[str, float] = Field(default_factory=dict)
    by_step: dict[str, float] = Field(default_factory=dict)
    line_items: list[CostLineItem] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    search_calls: int = 0
    search_results: int = 0
    images_generated: int = 0
    highest_cost_step: Optional[str] = None
    pricing: Optional[PricingConfig] = None
    notes: list[str] = Field(default_factory=list)


class RunEvent(BaseModel):
    kind: Literal["info", "node", "progress", "usage", "warning", "result", "error"]
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class BlogArtifacts(BaseModel):
    base_dir: Path
    markdown_path: Path
    run_json_path: Optional[Path] = None
    run_dir: Optional[Path] = None
    images_dir: Optional[Path] = None
    image_files: list[str] = Field(default_factory=list)


class SavedBlog(BaseModel):
    run_id: str
    source: Literal["run", "legacy"]
    title: str
    markdown_path: Path
    run_json_path: Optional[Path] = None
    run_dir: Optional[Path] = None
    base_dir: Path
    created_at: Optional[str] = None
    status: Literal["complete", "legacy"] = "complete"
    request_topic: Optional[str] = None
    provider_mix: list[str] = Field(default_factory=list)
    cost_summary: Optional[RunCostSummary] = None


class BlogRunResult(BaseModel):
    request: Optional[BlogRequest] = None
    blog_title: Optional[str] = None
    plan: Optional[Plan] = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    image_specs: list[ImageSpec] = Field(default_factory=list)
    final_markdown: str = ""
    artifacts: Optional[BlogArtifacts] = None
    events: list[RunEvent] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    resolved_runtime_config: Optional[ResolvedRuntimeConfig] = None
    provider_status_snapshot: list[ProviderStatus] = Field(default_factory=list)
    usage_records: list[UsageRecord] = Field(default_factory=list)
    cost_summary: Optional[RunCostSummary] = None


class BlogWorkflowState(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: list[str]
    evidence: list[EvidenceItem]
    plan: Optional[Plan]
    as_of: str
    recency_days: int
    enable_research: bool
    enable_images: bool
    sections: Annotated[list[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: list[dict[str, Any]]
    generated_images: list[dict[str, Any]]
    final: str
    warnings: Annotated[list[str], operator.add]
    usage_records: Annotated[list[dict[str, Any]], operator.add]
