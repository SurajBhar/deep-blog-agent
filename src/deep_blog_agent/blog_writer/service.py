"""Service layer for running the blog workflow."""

from __future__ import annotations

import os
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import BaseModel

from deep_blog_agent.artifacts.interfaces import ArtifactStore
from deep_blog_agent.artifacts.store import FileSystemArtifactStore
from deep_blog_agent.artifacts.utils import extract_title_from_markdown
from deep_blog_agent.blog_writer.contracts import (
    BlogRequest,
    BlogRunResult,
    BlogWorkflowState,
    EvidenceItem,
    ImageSpec,
    Plan,
    ProviderStatus,
    ResolvedRuntimeConfig,
    RunEvent,
    SessionRuntimeConfig,
    UsageRecord,
)
from deep_blog_agent.blog_writer.finops import calculate_cost_summary
from deep_blog_agent.blog_writer.graph import build_graph
from deep_blog_agent.core.errors import ArtifactStoreError, WorkflowExecutionError
from deep_blog_agent.core.runtime import resolve_runtime
from deep_blog_agent.core.serialization import to_jsonable
from deep_blog_agent.core.settings import AppSettings
from deep_blog_agent.providers.factory import ProviderBundle, build_default_provider_bundle


@dataclass
class BlogGenerationService:
    """Run and stream blog generation through a reusable service boundary."""

    settings: AppSettings
    providers: ProviderBundle
    artifact_store: ArtifactStore
    resolved_runtime_config: ResolvedRuntimeConfig | None = None
    provider_status_snapshot: list[ProviderStatus] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.graph = build_graph(self.providers)

    def run(self, request: BlogRequest) -> BlogRunResult:
        initial_state = self._build_initial_state(request)
        try:
            with self._runtime_environment():
                final_state = self.graph.invoke(initial_state)
        except Exception as exc:
            raise WorkflowExecutionError(f"Unable to run blog generation workflow: {exc}") from exc

        return self._finalize_result(
            request=request,
            final_state=final_state,
            events=[
                RunEvent(kind="info", message="Blog generation started."),
                RunEvent(kind="info", message="Blog generation completed."),
            ],
        )

    def stream(self, request: BlogRequest) -> Iterable[RunEvent]:
        initial_state = self._build_initial_state(request)
        current_state: dict[str, Any] = deepcopy(initial_state)
        events: list[RunEvent] = [RunEvent(kind="info", message="Blog generation started.")]
        emitted_warnings: set[str] = set()
        emitted_usage_count = 0

        yield events[0]

        try:
            with self._runtime_environment():
                for update in self.graph.stream(initial_state, stream_mode="updates"):
                    node_name, delta = self._split_update(update)
                    if node_name:
                        node_event = RunEvent(
                            kind="node",
                            message=f"Node: {node_name}",
                            payload={"name": node_name},
                        )
                        events.append(node_event)
                        yield node_event

                    current_state = self._merge_state(current_state, delta)

                    warnings = current_state.get("warnings", [])
                    for warning in warnings:
                        if warning in emitted_warnings:
                            continue
                        emitted_warnings.add(warning)
                        warning_event = RunEvent(kind="warning", message=warning)
                        events.append(warning_event)
                        yield warning_event

                    usage_records = current_state.get("usage_records", [])
                    if emitted_usage_count < len(usage_records):
                        for usage_record in usage_records[emitted_usage_count:]:
                            usage_event = RunEvent(
                                kind="usage",
                                message="Provider usage updated.",
                                payload={"usage": usage_record},
                            )
                            events.append(usage_event)
                            yield usage_event
                        emitted_usage_count = len(usage_records)

                    progress_event = RunEvent(
                        kind="progress",
                        message="Workflow progress updated.",
                        payload=self._state_summary(current_state),
                    )
                    events.append(progress_event)
                    yield progress_event
        except Exception as exc:
            error_event = RunEvent(kind="error", message=f"Workflow failed: {exc}")
            events.append(error_event)
            yield error_event
            raise WorkflowExecutionError(f"Unable to stream blog generation workflow: {exc}") from exc

        completed_event = RunEvent(kind="info", message="Blog generation completed.")
        events.append(completed_event)
        result = self._finalize_result(request=request, final_state=current_state, events=events)
        result_event = RunEvent(
            kind="result",
            message="Blog generation completed.",
            payload={"result": result.model_dump(mode="json")},
        )
        yield result_event

    def _build_initial_state(self, request: BlogRequest) -> BlogWorkflowState:
        return {
            "topic": request.topic.strip(),
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "as_of": request.as_of.isoformat(),
            "recency_days": 7,
            "enable_research": request.enable_research,
            "enable_images": request.enable_images,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "generated_images": [],
            "final": "",
            "warnings": [],
            "usage_records": [],
        }

    def _finalize_result(
        self,
        *,
        request: BlogRequest,
        final_state: dict[str, Any],
        events: list[RunEvent],
    ) -> BlogRunResult:
        plan = self._coerce_model(final_state.get("plan"), Plan)
        evidence = [self._coerce_model(item, EvidenceItem) for item in final_state.get("evidence", [])]
        image_specs = [self._coerce_model(item, ImageSpec) for item in final_state.get("image_specs", [])]
        usage_records = [self._coerce_usage_record(item) for item in final_state.get("usage_records", [])]
        markdown = final_state.get("final") or final_state.get("merged_md") or ""
        blog_title = (
            plan.blog_title
            if plan
            else extract_title_from_markdown(markdown, request.topic)
        )
        cost_summary = calculate_cost_summary(
            usage_records,
            self._resolved_runtime_config().pricing,
            markdown=markdown,
        )

        result = BlogRunResult(
            request=request,
            blog_title=blog_title,
            plan=plan,
            evidence=evidence,
            image_specs=image_specs,
            final_markdown=markdown,
            events=events,
            warnings=list(final_state.get("warnings", []) or []),
            resolved_runtime_config=self._resolved_runtime_config(),
            provider_status_snapshot=self.provider_status_snapshot,
            usage_records=usage_records,
            cost_summary=cost_summary,
        )

        try:
            artifacts = self.artifact_store.save_run(result, final_state.get("generated_images", []))
        except ArtifactStoreError:
            raise
        except Exception as exc:
            raise ArtifactStoreError(f"Unable to persist run artifacts: {exc}") from exc

        return result.model_copy(update={"artifacts": artifacts})

    @staticmethod
    def _coerce_model(value: Any, schema: type[BaseModel]):
        if isinstance(value, schema):
            return value
        if isinstance(value, dict):
            return schema(**value)
        return value

    @staticmethod
    def _coerce_usage_record(value: Any) -> UsageRecord:
        if isinstance(value, BaseModel):
            return value
        if isinstance(value, dict):
            usage_type = value.get("usage_type")
            if usage_type == "llm":
                from deep_blog_agent.blog_writer.contracts import LLMUsageRecord

                return LLMUsageRecord.model_validate(value)
            if usage_type == "search":
                from deep_blog_agent.blog_writer.contracts import SearchUsageRecord

                return SearchUsageRecord.model_validate(value)
            if usage_type == "image":
                from deep_blog_agent.blog_writer.contracts import ImageUsageRecord

                return ImageUsageRecord.model_validate(value)
        raise TypeError(f"Unsupported usage record: {value!r}")

    @staticmethod
    def _split_update(update: Any) -> tuple[str | None, dict[str, Any]]:
        if isinstance(update, dict) and len(update) == 1 and isinstance(next(iter(update.values())), dict):
            name = next(iter(update.keys()))
            return name, next(iter(update.values()))
        if isinstance(update, dict):
            return None, update
        return None, {}

    @staticmethod
    def _merge_state(current_state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
        merged = dict(current_state)
        for key, value in delta.items():
            if key in {"sections", "warnings", "usage_records"}:
                merged.setdefault(key, [])
                merged[key] = [*merged[key], *value]
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
        plan = state.get("plan")
        task_count = None
        outline = None
        if isinstance(plan, Plan):
            task_count = len(plan.tasks)
            outline = {
                "blog_title": plan.blog_title,
                "audience": plan.audience,
                "tone": plan.tone,
                "blog_kind": plan.blog_kind,
                "constraints": list(plan.constraints),
                "tasks": [
                    {
                        "id": task.id,
                        "title": task.title,
                        "goal": task.goal,
                        "bullets": list(task.bullets),
                        "target_words": task.target_words,
                        "requires_research": task.requires_research,
                        "requires_citations": task.requires_citations,
                    }
                    for task in plan.tasks
                ],
            }
        elif isinstance(plan, dict):
            task_count = len(plan.get("tasks", []))
            outline = to_jsonable(plan)

        sections = state.get("sections") or []
        ordered_sections = [markdown for _task_id, markdown in sorted(sections, key=lambda item: item[0])]
        blog_title = (
            outline.get("blog_title")
            if isinstance(outline, dict)
            else None
        ) or state.get("topic") or "Draft"
        draft_preview = state.get("final") or state.get("merged_md") or ""
        if not draft_preview and ordered_sections:
            draft_preview = f"# {blog_title}\n\n" + "\n\n".join(ordered_sections).strip()
        if len(draft_preview) > 8000:
            draft_preview = draft_preview[:8000].rstrip() + "\n\n[Draft preview truncated]"

        evidence_preview = []
        for item in state.get("evidence") or []:
            if isinstance(item, EvidenceItem):
                evidence_preview.append(item.model_dump(mode="json"))
            elif isinstance(item, dict):
                evidence_preview.append(
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "published_at": item.get("published_at"),
                        "source": item.get("source"),
                    }
                )

        image_specs_preview = []
        for item in state.get("image_specs") or []:
            if isinstance(item, ImageSpec):
                image_specs_preview.append(item.model_dump(mode="json"))
            elif isinstance(item, dict):
                image_specs_preview.append(
                    {
                        "filename": item.get("filename"),
                        "caption": item.get("caption"),
                        "alt": item.get("alt"),
                        "size": item.get("size"),
                        "quality": item.get("quality"),
                    }
                )

        return to_jsonable(
            {
                "mode": state.get("mode"),
                "needs_research": state.get("needs_research"),
                "queries": (state.get("queries") or [])[:5],
                "evidence_count": len(state.get("evidence") or []),
                "evidence_preview": evidence_preview[:8],
                "tasks": task_count,
                "outline": outline,
                "images": len(state.get("image_specs") or []),
                "image_specs_preview": image_specs_preview[:6],
                "sections_done": len(state.get("sections") or []),
                "draft_preview": draft_preview,
                "warnings": len(state.get("warnings") or []),
                "usage_records": len(state.get("usage_records") or []),
            }
        )

    def _resolved_runtime_config(self) -> ResolvedRuntimeConfig:
        if self.resolved_runtime_config is not None:
            return self.resolved_runtime_config
        return ResolvedRuntimeConfig(
            openai_model=self.settings.openai_model,
            google_image_model=self.settings.google_image_model,
            default_enable_research=self.settings.default_enable_research,
            default_enable_images=self.settings.default_enable_images,
            langsmith_tracing=self.settings.langsmith_tracing,
            langsmith_project=self.settings.langsmith_project,
            pricing=self.settings.pricing,
            credential_sources={},
        )

    @contextmanager
    def _runtime_environment(self):
        runtime_config = self._resolved_runtime_config()
        tracked_names = ["LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_PROJECT"]
        previous_values = {name: os.environ.get(name) for name in tracked_names}

        if self.settings.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = self.settings.langsmith_api_key
        else:
            os.environ.pop("LANGSMITH_API_KEY", None)

        if runtime_config.langsmith_tracing:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
        else:
            os.environ.pop("LANGSMITH_TRACING", None)
            os.environ.pop("LANGCHAIN_TRACING_V2", None)

        if runtime_config.langsmith_project:
            os.environ["LANGSMITH_PROJECT"] = runtime_config.langsmith_project
        else:
            os.environ.pop("LANGSMITH_PROJECT", None)

        try:
            yield
        finally:
            for name, previous in previous_values.items():
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous


def build_blog_generation_service(
    settings: AppSettings,
    overrides: SessionRuntimeConfig | None = None,
    *,
    validate_providers: bool = False,
) -> BlogGenerationService:
    """Build a configured service using deployment settings plus optional session overrides."""

    resolved_runtime = resolve_runtime(settings, overrides, validate_providers=validate_providers)
    providers = build_default_provider_bundle(resolved_runtime.settings)
    artifact_store = FileSystemArtifactStore(resolved_runtime.settings)
    return BlogGenerationService(
        settings=resolved_runtime.settings,
        providers=providers,
        artifact_store=artifact_store,
        resolved_runtime_config=resolved_runtime.config,
        provider_status_snapshot=resolved_runtime.provider_statuses,
    )


def build_default_blog_generation_service(settings: AppSettings | None = None) -> BlogGenerationService:
    """Build the default configured service for the app."""

    runtime_settings = settings or AppSettings.load()
    return build_blog_generation_service(runtime_settings)
