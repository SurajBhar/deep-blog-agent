"""Node implementations for the blog writer workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from langgraph.types import Send

from deep_blog_agent.blog_writer.contracts import (
    BlogWorkflowState,
    EvidenceItem,
    EvidencePack,
    GlobalImagePlan,
    ImageSpec,
    Plan,
    PromptMessage,
    RouterDecision,
    SearchResult,
    Task,
)
from deep_blog_agent.blog_writer.prompts import (
    DECIDE_IMAGES_SYSTEM,
    ORCH_SYSTEM,
    RESEARCH_SYSTEM,
    ROUTER_SYSTEM,
    WORKER_SYSTEM,
)
from deep_blog_agent.core.errors import ImageGenerationError, ProviderConfigurationError, SearchProviderError
from deep_blog_agent.providers.factory import ProviderBundle


def _iso_to_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def route_next(state: BlogWorkflowState) -> str:
    if state.get("needs_research") and state.get("enable_research", True):
        return "research"
    return "orchestrator"


def _usage_record(record, *, step: str, **metadata: Any) -> list[dict[str, Any]]:
    if record is None:
        return []

    merged_metadata = dict(record.metadata)
    merged_metadata.update(metadata)
    updated = record.model_copy(update={"step": step, "metadata": merged_metadata})
    return [updated.model_dump(mode="json")]


@dataclass
class BlogWriterNodes:
    """Collection of workflow nodes using injected dependencies."""

    providers: ProviderBundle

    def router_node(self, state: BlogWorkflowState) -> dict[str, Any]:
        if not state.get("enable_research", True):
            return {
                "needs_research": False,
                "mode": "closed_book",
                "queries": [],
                "recency_days": 3650,
                "usage_records": [],
            }

        decision_result = self.providers.llm.invoke_structured(
            [
                PromptMessage(role="system", content=ROUTER_SYSTEM),
                PromptMessage(
                    role="user",
                    content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}",
                ),
            ],
            RouterDecision,
        )
        decision = decision_result.value

        if decision.mode == "open_book":
            recency_days = 7
        elif decision.mode == "hybrid":
            recency_days = 45
        else:
            recency_days = 3650

        return {
            "needs_research": decision.needs_research,
            "mode": decision.mode,
            "queries": decision.queries,
            "recency_days": recency_days,
            "usage_records": _usage_record(decision_result.usage, step="router"),
        }

    def research_node(self, state: BlogWorkflowState) -> dict[str, Any]:
        queries = (state.get("queries") or [])[:10]
        warnings: list[str] = []
        raw_results: list[SearchResult] = []
        usage_records: list[dict[str, Any]] = []

        for query in queries:
            try:
                search_result = self.providers.search.search(query, max_results=6)
                raw_results.extend(search_result.results)
                usage_records.extend(_usage_record(search_result.usage, step="research_search"))
            except ProviderConfigurationError as exc:
                warnings.append(str(exc))
                break
            except SearchProviderError as exc:
                warnings.append(f"Search failed for '{query}': {exc}")

        if not raw_results:
            return {"evidence": [], "warnings": warnings, "usage_records": usage_records}

        pack_result = self.providers.llm.invoke_structured(
            [
                PromptMessage(role="system", content=RESEARCH_SYSTEM),
                PromptMessage(
                    role="user",
                    content=(
                        f"As-of date: {state['as_of']}\n"
                        f"Recency days: {state['recency_days']}\n\n"
                        f"Raw results:\n{[result.model_dump() for result in raw_results]}"
                    ),
                ),
            ],
            EvidencePack,
        )
        pack = pack_result.value

        deduped: dict[str, EvidenceItem] = {}
        for evidence in pack.evidence:
            if evidence.url:
                deduped[evidence.url] = evidence
        filtered = list(deduped.values())

        if state.get("mode") == "open_book":
            as_of = date.fromisoformat(state["as_of"])
            cutoff = as_of - timedelta(days=int(state["recency_days"]))
            filtered = [
                evidence
                for evidence in filtered
                if (published := _iso_to_date(evidence.published_at)) and published >= cutoff
            ]

        usage_records.extend(_usage_record(pack_result.usage, step="research_synthesis"))
        return {"evidence": filtered, "warnings": warnings, "usage_records": usage_records}

    def orchestrator_node(self, state: BlogWorkflowState) -> dict[str, Any]:
        mode = state.get("mode", "closed_book")
        evidence = state.get("evidence", [])
        forced_kind = "news_roundup" if mode == "open_book" else None

        plan_result = self.providers.llm.invoke_structured(
            [
                PromptMessage(role="system", content=ORCH_SYSTEM),
                PromptMessage(
                    role="user",
                    content=(
                        f"Topic: {state['topic']}\n"
                        f"Mode: {mode}\n"
                        f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                        f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                        f"Evidence:\n{[item.model_dump() for item in evidence][:16]}"
                    ),
                ),
            ],
            Plan,
        )
        plan = plan_result.value
        if forced_kind:
            plan.blog_kind = "news_roundup"

        return {"plan": plan, "usage_records": _usage_record(plan_result.usage, step="planning")}

    def fanout(self, state: BlogWorkflowState):
        assert state["plan"] is not None
        return [
            Send(
                "worker",
                {
                    "task": task.model_dump(),
                    "topic": state["topic"],
                    "mode": state["mode"],
                    "as_of": state["as_of"],
                    "recency_days": state["recency_days"],
                    "plan": state["plan"].model_dump(),
                    "evidence": [item.model_dump() for item in state.get("evidence", [])],
                },
            )
            for task in state["plan"].tasks
        ]

    def worker_node(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = Task(**payload["task"])
        plan = Plan(**payload["plan"])
        evidence = [EvidenceItem(**item) for item in payload.get("evidence", [])]

        bullets_text = "\n- " + "\n- ".join(task.bullets)
        evidence_text = "\n".join(
            f"- {item.title} | {item.url} | {item.published_at or 'date:unknown'}"
            for item in evidence[:20]
        )

        section_result = self.providers.llm.invoke(
            [
                PromptMessage(role="system", content=WORKER_SYSTEM),
                PromptMessage(
                    role="user",
                    content=(
                        f"Blog title: {plan.blog_title}\n"
                        f"Audience: {plan.audience}\n"
                        f"Tone: {plan.tone}\n"
                        f"Blog kind: {plan.blog_kind}\n"
                        f"Constraints: {plan.constraints}\n"
                        f"Topic: {payload['topic']}\n"
                        f"Mode: {payload.get('mode')}\n"
                        f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                        f"Section title: {task.title}\n"
                        f"Goal: {task.goal}\n"
                        f"Target words: {task.target_words}\n"
                        f"Tags: {task.tags}\n"
                        f"requires_research: {task.requires_research}\n"
                        f"requires_citations: {task.requires_citations}\n"
                        f"requires_code: {task.requires_code}\n"
                        f"Bullets:{bullets_text}\n\n"
                        f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                    ),
                ),
            ]
        )
        section_markdown = section_result.text.strip()

        return {
            "sections": [(task.id, section_markdown)],
            "usage_records": _usage_record(section_result.usage, step="write_section", task_id=task.id, task_title=task.title),
        }

    @staticmethod
    def merge_content(state: BlogWorkflowState) -> dict[str, Any]:
        plan = state["plan"]
        if plan is None:
            raise ValueError("merge_content called without a plan.")
        ordered_sections = [markdown for _, markdown in sorted(state["sections"], key=lambda item: item[0])]
        body = "\n\n".join(ordered_sections).strip()
        merged_markdown = f"# {plan.blog_title}\n\n{body}\n"
        return {"merged_md": merged_markdown}

    def decide_images(self, state: BlogWorkflowState) -> dict[str, Any]:
        if not state.get("enable_images", True):
            return {"md_with_placeholders": state["merged_md"], "image_specs": [], "usage_records": []}

        plan = state["plan"]
        assert plan is not None
        image_plan_result = self.providers.llm.invoke_structured(
            [
                PromptMessage(role="system", content=DECIDE_IMAGES_SYSTEM),
                PromptMessage(
                    role="user",
                    content=(
                        f"Blog kind: {plan.blog_kind}\n"
                        f"Topic: {state['topic']}\n\n"
                        "Insert placeholders + propose image prompts.\n\n"
                        f"{state['merged_md']}"
                    ),
                ),
            ],
            GlobalImagePlan,
        )
        image_plan = image_plan_result.value

        return {
            "md_with_placeholders": image_plan.md_with_placeholders,
            "image_specs": [image.model_dump() for image in image_plan.images],
            "usage_records": _usage_record(image_plan_result.usage, step="plan_images"),
        }

    def generate_and_place_images(self, state: BlogWorkflowState) -> dict[str, Any]:
        markdown = state.get("md_with_placeholders") or state["merged_md"]
        image_specs = [ImageSpec(**item) for item in state.get("image_specs", []) or []]
        warnings: list[str] = []
        generated_images: list[dict[str, Any]] = []
        usage_records: list[dict[str, Any]] = []

        if not image_specs:
            return {"final": markdown, "generated_images": [], "warnings": warnings, "usage_records": usage_records}

        for spec in image_specs:
            try:
                image_result = self.providers.image.generate_image(
                    spec.prompt,
                    size=spec.size,
                    quality=spec.quality,
                )
                image_bytes = image_result.image_bytes
                generated_images.append(
                    {
                        "filename": spec.filename,
                        "alt": spec.alt,
                        "caption": spec.caption,
                        "bytes": image_bytes,
                    }
                )
                markdown = markdown.replace(
                    spec.placeholder,
                    f"![{spec.alt}](images/{spec.filename})\n*{spec.caption}*",
                )
                usage_records.extend(
                    _usage_record(
                        image_result.usage,
                        step="generate_images",
                        asset_name=spec.filename,
                        alt=spec.alt,
                    )
                )
            except (ProviderConfigurationError, ImageGenerationError) as exc:
                warnings.append(f"Image generation failed for '{spec.filename}': {exc}")
                markdown = markdown.replace(spec.placeholder, self._image_failure_block(spec, str(exc)))

        return {
            "final": markdown,
            "generated_images": generated_images,
            "warnings": warnings,
            "usage_records": usage_records,
        }

    @staticmethod
    def _image_failure_block(spec: ImageSpec, error: str) -> str:
        return (
            f"> **[IMAGE GENERATION FAILED]** {spec.caption}\n>\n"
            f"> **Alt:** {spec.alt}\n>\n"
            f"> **Prompt:** {spec.prompt}\n>\n"
            f"> **Error:** {error}\n"
        )
