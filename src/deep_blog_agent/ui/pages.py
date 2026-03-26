"""Multipage Streamlit UI for Deep Blog Agent."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Iterable, Sequence

import pandas as pd
import streamlit as st

from deep_blog_agent.artifacts.interfaces import ArtifactStore
from deep_blog_agent.artifacts.utils import extract_title_from_markdown, slugify_title
from deep_blog_agent.blog_writer.contracts import (
    BlogRequest,
    BlogRunResult,
    EvidenceItem,
    PricingConfig,
    ProviderStatus,
    ResolvedRuntimeConfig,
    RunEvent,
    SavedBlog,
    SessionRuntimeConfig,
)
from deep_blog_agent.blog_writer.service import build_blog_generation_service
from deep_blog_agent.core.errors import ArtifactStoreError, WorkflowExecutionError
from deep_blog_agent.core.runtime import ResolvedRuntime, resolve_runtime
from deep_blog_agent.core.settings import AppSettings
from . import session
from .components import (
    MetricCardData,
    TimelineStep,
    render_badge_row,
    render_empty_state,
    render_key_value_grid,
    render_log_panel,
    render_metric_cards,
    render_timeline,
)
from .renderers import build_renderable_blog_markdown, coerce_plan_dict, render_markdown_with_local_images
from .theme import render_page_header, render_section_intro, render_status_strip
from .view_models import (
    aggregate_provider_costs,
    extract_queries_from_events,
    finops_rows,
    format_duration,
    format_int,
    format_timestamp,
    format_usd,
    history_rows,
    summarize_result,
    summarize_saved_blogs,
)

_PROMPT_EXAMPLES = [
    {
        "category": "Technical explanation",
        "title": "RAG evaluation in production",
        "prompt": (
            "Write a technical blog for senior ML engineers explaining how retrieval-augmented generation systems are "
            "evaluated in production. Cover retrieval quality, answer quality, human review, offline benchmarks, and "
            "failure analysis. Include a pragmatic section on tradeoffs and operating metrics."
        ),
    },
    {
        "category": "Architecture deep dive",
        "title": "LangGraph writing workflow",
        "prompt": (
            "Describe how LangGraph supports long-running technical writing workflows. Explain graph state, retries, "
            "checkpointing, human review points, and how research, outlining, drafting, and revision can be modeled "
            "as distinct nodes."
        ),
    },
    {
        "category": "Comparison article",
        "title": "MCP vs traditional tool calling",
        "prompt": (
            "Write a comparison article for platform engineers on MCP tool integration versus traditional API tool "
            "calling. Compare ergonomics, safety boundaries, observability, latency, and deployment implications. Use "
            "clear examples and a recommendation matrix."
        ),
    },
    {
        "category": "Tutorial-style prompt",
        "title": "Build an agentic blog pipeline",
        "prompt": (
            "Create a tutorial-style blog post that walks experienced Python developers through building an agentic "
            "blog generation pipeline with research, planning, section drafting, and optional image generation. Include "
            "architecture, code snippets, and common failure modes."
        ),
    },
    {
        "category": "Research summary",
        "title": "Recent agent orchestration patterns",
        "prompt": (
            "Produce a research summary blog on recent agent orchestration patterns in production AI systems. Focus "
            "on state machines, graph orchestration, tool reliability, cost control, and traceability. Include current "
            "examples, source-backed claims, and a short section on open questions."
        ),
    },
]
_STAGE_ORDER = [
    "router",
    "research",
    "orchestrator",
    "worker",
    "merge_content",
    "decide_images",
    "generate_and_place_images",
]
_STAGE_LABELS = {
    "router": "Assess request",
    "research": "Research sources",
    "orchestrator": "Plan outline",
    "worker": "Draft article",
    "merge_content": "Assemble draft",
    "decide_images": "Plan images",
    "generate_and_place_images": "Finalize article",
}


@dataclass(frozen=True)
class UIContext:
    base_settings: AppSettings
    resolved_runtime: ResolvedRuntime
    artifact_store: ArtifactStore


def render_home_page(context: UIContext) -> None:
    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    st.markdown("### Technical Content Writer Agent")
    st.caption("Technical research -> draft generation")

    if not any(status.provider == "openai" and status.ready for status in context.resolved_runtime.provider_statuses):
        st.warning("OpenAI credentials are missing. Add them in Settings before starting a run.")

    result = session.get_last_result()
    composer_container = st.container()
    submitted = False
    topic = session.get_blog_prompt().strip()

    with composer_container:
        st.caption("PROMPT")
        composer_columns = st.columns([5, 1.2])
        composer_columns[0].text_area(
            "Describe the next blog run",
            key=session.BLOG_PROMPT_KEY,
            height=62,
            label_visibility="collapsed",
            placeholder=(
                "Write a technical blog for platform engineers on the tradeoffs of running "
                "retrieval-augmented generation systems in production."
            ),
        )
        submitted = composer_columns[1].button(
            "Generate blog",
            type="primary",
            use_container_width=True,
            key="home-generate-blog",
        )
        st.caption("Describe the topic, audience, depth, and constraints.")
        topic = session.get_blog_prompt().strip()
        if submitted and not topic:
            st.warning("Enter a prompt before starting a run.")

    workflow_container = st.empty()
    output_container = st.empty()

    if submitted and topic:
        session.reset_logs()
        runtime_overrides = session.get_runtime_overrides()
        runtime_service = build_blog_generation_service(context.base_settings, runtime_overrides)
        request = BlogRequest(
            topic=topic,
            as_of=st.session_state[session.AS_OF_INPUT_KEY],
            enable_research=st.session_state[session.CREATE_ENABLE_RESEARCH_KEY],
            enable_images=st.session_state[session.CREATE_ENABLE_IMAGES_KEY],
            runtime_overrides=runtime_overrides,
        )
        new_result = _run_generation(
            request,
            runtime_service,
            workflow_container=workflow_container,
            output_container=output_container,
            artifact_store=context.artifact_store,
        )
        if new_result:
            result = new_result
    else:
        with workflow_container.container():
            if result:
                _render_execution_summary(result)
            else:
                _render_idle_workflow_workspace()

        with output_container.container():
            st.divider()
            if not result:
                _render_idle_result_workspace()
            else:
                _render_result_workspace(result, context.artifact_store, show_open_detail=True)


def render_prompt_examples_page(context: UIContext) -> None:
    render_page_header(
        "Prompt Examples",
        "Use these examples to understand how detailed, structured prompts produce stronger blog drafts.",
    )
    with st.container(border=True):
        st.markdown("**What makes a good prompt**")
        st.caption(
            "Strong prompts name the topic, target reader, desired depth, structure, and any constraints on tone, sources, or code."
        )
        st.caption("Load any example into Home, then adapt it to your audience, depth, and constraints.")

    for index, example in enumerate(_PROMPT_EXAMPLES):
        with st.container(border=True):
            st.caption(example["category"].upper())
            st.markdown(f"**{example['title']}**")
            st.code(example["prompt"], language="markdown")
            if st.button("Use on Home", key=f"prompt-example-{index}"):
                _open_home_with_prompt(example["prompt"])


def render_run_history_page(context: UIContext) -> None:
    render_page_header(
        "Run History",
        "Browse previous research and writing sessions, then reopen them in Run Detail or reuse the original prompt on Home.",
    )
    filter_columns = st.columns([2, 1])
    search_text = filter_columns[0].text_input("Search sessions", placeholder="Filter by title or topic")
    source_filter = filter_columns[1].selectbox("Source", options=["all", "run", "legacy"], index=0)

    saved_runs = context.artifact_store.list_runs(limit=200, search_text=search_text)
    if source_filter != "all":
        saved_runs = [saved_blog for saved_blog in saved_runs if saved_blog.source == source_filter]

    if not saved_runs:
        render_empty_state("No matching sessions", "Try a different search or generate a new blog run.")
        return

    render_section_intro("Sessions", "Saved sessions", "Each row is a previous writing session.")
    for saved_blog in saved_runs:
        _render_saved_run_row(saved_blog, context.artifact_store, card_prefix="history")

    with st.expander("Show table view"):
        frame = pd.DataFrame(history_rows(saved_runs))
        frame = frame[
            ["created_at_label", "title", "status", "source", "topic", "providers", "estimated_cost_label"]
        ]
        st.dataframe(frame, hide_index=True)


def render_run_detail_page(context: UIContext) -> None:
    saved_runs = context.artifact_store.list_runs(limit=200)
    render_page_header(
        "Run Detail",
        "Open one saved session and review the prompt, generated draft, sources, assets, and trace output.",
    )

    if not saved_runs and not session.get_last_result():
        render_empty_state("No session selected", "Generate a new blog or load one from Run History.")
        return

    selected_blog, missing_selection = _resolve_selected_blog(saved_runs)
    if saved_runs:
        labels = [f"{item.title} · {format_timestamp(item.created_at)}" for item in saved_runs]
        selected_index = saved_runs.index(selected_blog) if selected_blog else 0
        selected_label = st.selectbox("Saved session", options=labels, index=selected_index)
        selected_blog = saved_runs[labels.index(selected_label)]
        session.set_selected_run_id(selected_blog.run_id)
        if missing_selection:
            st.error("The selected session could not be found. Showing the most recent available session instead.")

    result = _resolve_selected_result(context.artifact_store, selected_blog)
    if not result:
        render_empty_state("Run unavailable", "The selected run could not be loaded from disk.")
        return

    summary = summarize_result(result)
    render_metric_cards(
        [
            MetricCardData("Words", format_int(summary["word_count"])),
            MetricCardData("Evidence", format_int(summary["evidence_count"])),
            MetricCardData("Tokens", format_int(summary["total_tokens"])),
            MetricCardData("Estimated cost", format_usd(summary["estimated_cost"])),
        ],
        columns=4,
    )

    with st.expander("Request and runtime", expanded=True):
        request = result.request
        runtime = result.resolved_runtime_config
        updated_at = _result_updated_timestamp(result)
        render_key_value_grid(
            [
                ("Prompt", request.topic if request else "Unknown"),
                ("As-of date", request.as_of.isoformat() if request else "Unknown"),
                ("Research", "Enabled" if request and request.enable_research else "Disabled"),
                ("Images", "Enabled" if request and request.enable_images else "Disabled"),
                ("OpenAI model", runtime.openai_model if runtime else "Unknown"),
                ("Image model", runtime.google_image_model if runtime else "Unknown"),
                ("Run ID", session.get_result_run_id(result) or "Unknown"),
                ("Updated", format_timestamp(updated_at) if updated_at else "Unknown"),
            ],
            columns=2,
        )

    _render_result_workspace(result, context.artifact_store, show_open_detail=False, third_action="bundle")


def render_settings_page(context: UIContext) -> None:
    render_page_header(
        "Settings",
        "Session-scoped configuration for credentials, models, tracing, and pricing assumptions.",
    )
    st.caption("Session credentials stay in this browser session only and are not written to saved artifacts.")

    overrides = session.get_runtime_overrides()
    _sync_settings_widgets(context.resolved_runtime.config, overrides)
    validation_statuses = context.resolved_runtime.provider_statuses

    tabs = st.tabs(["Credentials", "Defaults", "Pricing"])
    with tabs[0]:
        st.text_input("OPENAI_API_KEY", type="password", key=session.SETTINGS_OPENAI_KEY_KEY)
        st.text_input("TAVILY_API_KEY", type="password", key=session.SETTINGS_TAVILY_KEY_KEY)
        st.text_input("GOOGLE_API_KEY", type="password", key=session.SETTINGS_GOOGLE_KEY_KEY)
        st.text_input("LANGSMITH_API_KEY", type="password", key=session.SETTINGS_LANGSMITH_API_KEY)

    with tabs[1]:
        st.text_input("OPENAI_MODEL", key=session.SETTINGS_OPENAI_MODEL_KEY)
        st.text_input("GOOGLE_IMAGE_MODEL", key=session.SETTINGS_GOOGLE_IMAGE_MODEL_KEY)
        toggle_columns = st.columns(2)
        toggle_columns[0].toggle("Default research enabled", key=session.SETTINGS_ENABLE_RESEARCH_KEY)
        toggle_columns[1].toggle("Default images enabled", key=session.SETTINGS_ENABLE_IMAGES_KEY)
        tracing_columns = st.columns(2)
        tracing_columns[0].toggle("LangSmith tracing", key=session.SETTINGS_LANGSMITH_TRACING_KEY)
        tracing_columns[1].text_input("LangSmith project", key=session.SETTINGS_LANGSMITH_PROJECT_KEY)

    with tabs[2]:
        st.text_input("Pricing label", key=session.SETTINGS_PRICING_LABEL_KEY)
        price_columns = st.columns(2)
        price_columns[0].number_input(
            "OpenAI input price / 1M tokens (USD)",
            min_value=0.0,
            step=0.01,
            key=session.SETTINGS_OPENAI_INPUT_PRICE_KEY,
            format="%.6f",
        )
        price_columns[1].number_input(
            "OpenAI output price / 1M tokens (USD)",
            min_value=0.0,
            step=0.01,
            key=session.SETTINGS_OPENAI_OUTPUT_PRICE_KEY,
            format="%.6f",
        )
        price_columns = st.columns(2)
        price_columns[0].number_input(
            "Tavily price / query (USD)",
            min_value=0.0,
            step=0.001,
            key=session.SETTINGS_TAVILY_QUERY_PRICE_KEY,
            format="%.6f",
        )
        price_columns[1].number_input(
            "Google image price / image (USD)",
            min_value=0.0,
            step=0.001,
            key=session.SETTINGS_GOOGLE_IMAGE_PRICE_KEY,
            format="%.6f",
        )

    action_columns = st.columns(3)
    if action_columns[0].button("Apply to session", type="primary", key="settings-apply"):
        runtime_overrides = _build_runtime_overrides_from_widgets()
        session.set_runtime_overrides(runtime_overrides)
        st.session_state[session.CREATE_ENABLE_RESEARCH_KEY] = bool(runtime_overrides.default_enable_research)
        st.session_state[session.CREATE_ENABLE_IMAGES_KEY] = bool(runtime_overrides.default_enable_images)
        st.success("Session overrides applied.")
        st.rerun()

    if action_columns[1].button("Clear overrides", key="settings-clear"):
        session.clear_runtime_overrides()
        _reset_settings_widgets(context.base_settings)
        st.session_state[session.CREATE_ENABLE_RESEARCH_KEY] = context.base_settings.default_enable_research
        st.session_state[session.CREATE_ENABLE_IMAGES_KEY] = context.base_settings.default_enable_images
        st.rerun()

    if action_columns[2].button("Validate providers", key="settings-validate"):
        validation_statuses = resolve_runtime(
            context.base_settings,
            _build_runtime_overrides_from_widgets(),
            validate_providers=True,
        ).provider_statuses

    with st.expander("Provider readiness"):
        st.dataframe(pd.DataFrame(_provider_rows(validation_statuses)), hide_index=True)


def render_finops_page(context: UIContext) -> None:
    saved_runs = context.artifact_store.list_cost_history(limit=200)
    render_page_header(
        "FinOps",
        "A lightweight view of spend, usage, and the runs that cost the most.",
    )
    if not saved_runs:
        render_empty_state("No cost data", "Saved runs with usage records will appear here.")
        return

    summary = summarize_saved_blogs(saved_runs)
    render_metric_cards(
        [
            MetricCardData("Total runs", format_int(summary["total_runs"])),
            MetricCardData("Runs with cost", format_int(summary["runs_with_cost"])),
            MetricCardData("Total spend", format_usd(summary["total_cost"])),
            MetricCardData("Average cost", format_usd(summary["average_cost"])),
        ],
        columns=4,
    )

    rows = finops_rows(saved_runs)
    frame = pd.DataFrame(rows)
    trend_frame = frame.copy()
    trend_frame["created_at"] = pd.to_datetime(trend_frame["created_at"], errors="coerce")
    trend_frame = trend_frame.dropna(subset=["created_at", "estimated_cost_usd"]).sort_values("created_at")
    if not trend_frame.empty:
        st.line_chart(trend_frame.set_index("created_at")["estimated_cost_usd"])

    provider_totals = aggregate_provider_costs(saved_runs)
    if provider_totals:
        st.bar_chart(pd.Series(provider_totals))

    display_frame = frame[
        [
            "created_at_label",
            "title",
            "estimated_cost_usd",
            "cost_per_1000_words_usd",
            "total_tokens",
            "search_calls",
            "images_generated",
            "highest_cost_step",
        ]
    ]
    st.dataframe(display_frame, hide_index=True)


def _render_idle_workflow_workspace() -> None:
    st.caption("LIVE WORKFLOW")
    st.subheader("Agent activity")
    st.caption("The agent's research, planning, and writing steps will stream here during execution.")
    render_empty_state(
        "Start a run to see the research workflow unfold here.",
        "The agent will surface research, planning, drafting, and finalization updates as they happen.",
        ghost_steps=["Research", "Plan", "Draft", "Refine", "Finalize"],
    )


def _render_idle_result_workspace() -> None:
    st.caption("GENERATED RESULT")
    st.subheader("Latest draft")
    st.caption("The generated article will appear here once the agent completes the draft.")
    render_empty_state(
        "The generated article will appear here once the agent completes the draft.",
        "Outline, sources, images, and trace output stay attached to the blog in this workspace.",
        ghost_steps=["Blog", "Outline", "Sources", "Images", "Trace"],
    )


def _render_result_workspace(
    result: BlogRunResult,
    artifact_store: ArtifactStore,
    *,
    show_open_detail: bool,
    third_action: str = "open_detail",
) -> None:
    article_markdown = build_renderable_blog_markdown(result.final_markdown, result.evidence)
    blog_title = result.blog_title or extract_title_from_markdown(article_markdown, "blog")
    updated_at = _result_updated_timestamp(result)
    run_id = session.get_result_run_id(result)

    st.caption("GENERATED RESULT")
    st.subheader("Generated Blog")
    st.markdown(f"**{blog_title}**")
    meta_parts = ["Complete"]
    if updated_at:
        meta_parts.append(f"Updated {format_timestamp(updated_at)}")
    if run_id:
        meta_parts.append(f"Run {run_id}")
    st.caption(" • ".join(meta_parts))
    _render_result_actions(
        markdown=article_markdown,
        title=blog_title,
        prompt=result.request.topic if result.request else None,
        result=result,
        artifact_store=artifact_store,
        show_open_detail=show_open_detail,
        third_action=third_action,
        action_key_prefix=f"result-{run_id or slugify_title(blog_title)}",
    )

    tabs = st.tabs(["Blog", "Outline", "Sources", "Images", "Trace"])
    with tabs[0]:
        _render_blog_tab(article_markdown, result)
    with tabs[1]:
        _render_outline_tab(coerce_plan_dict(result), fallback_title=blog_title)
    with tabs[2]:
        _render_sources_tab(result.evidence, extract_queries_from_events(result.events))
    with tabs[3]:
        _render_images_tab(result, artifact_store)
    with tabs[4]:
        _render_trace_tab(result)


def _render_execution_summary(result: BlogRunResult) -> None:
    current_node, seen_nodes, progress_summary = _extract_execution_state(result.events)
    request = result.request
    updated_at = _result_updated_timestamp(result)
    _render_workflow_panel(
        current_node=current_node,
        seen_nodes=seen_nodes,
        progress_summary=progress_summary,
        enable_research=bool(request.enable_research) if request else True,
        enable_images=bool(request.enable_images) if request else True,
        warnings=result.warnings,
        log_lines=[_format_log_line(event) for event in result.events[-80:]],
        is_complete=True,
        context_label=f"Last updated {format_timestamp(updated_at)}" if updated_at else None,
    )


def _render_workflow_panel(
    *,
    current_node: str | None,
    seen_nodes: Sequence[str],
    progress_summary: dict[str, object],
    enable_research: bool,
    enable_images: bool,
    warnings: Sequence[str],
    log_lines: Sequence[str],
    is_complete: bool,
    elapsed_seconds: float | None = None,
    context_label: str | None = None,
) -> None:
    st.caption("LIVE WORKFLOW")
    st.subheader("Agent activity")
    status_label = _workflow_status_label(current_node, is_complete=is_complete)
    status_state = "complete" if is_complete else "running"
    progress_value = _estimate_execution_progress(current_node, progress_summary, is_complete=is_complete)
    context_parts = [_STAGE_LABELS.get(current_node or "", "Waiting for the next run")]
    if elapsed_seconds is not None:
        context_parts.append(f"Elapsed {format_duration(elapsed_seconds)}")
    elif context_label:
        context_parts.append(context_label)

    with st.status(status_label, state=status_state, expanded=True):
        st.caption(" • ".join(part for part in context_parts if part))
        st.progress(progress_value, text=_execution_progress_label(current_node, progress_summary, is_complete=is_complete))
        render_status_strip(
            _build_stage_tracker(
                current_node,
                seen_nodes,
                progress_summary,
                enable_research=enable_research,
                enable_images=enable_images,
                is_complete=is_complete,
            )
        )

        tasks = int(progress_summary.get("tasks") or 0)
        sections_done = min(int(progress_summary.get("sections_done") or 0), tasks) if tasks else 0
        render_badge_row(
            [
                (f"Queries {len(progress_summary.get('queries') or [])}", "neutral"),
                (f"Sources {int(progress_summary.get('evidence_count') or 0)}", "neutral"),
                (f"Sections {sections_done}/{tasks}" if tasks else "Sections pending", "neutral"),
                (f"Images {'on' if enable_images else 'off'}", "neutral"),
            ]
        )
        st.markdown("**Activity stream**")
        render_timeline(
            _build_execution_timeline(
                current_node,
                seen_nodes,
                progress_summary,
                enable_research=enable_research,
                enable_images=enable_images,
                is_complete=is_complete,
            ),
            empty_message="The agent will begin streaming activity as soon as the run starts.",
        )

        if warnings:
            for warning in warnings[-2:]:
                st.warning(warning)

        with st.expander("Detailed trace", expanded=False):
            render_log_panel(log_lines[-120:], empty_message="Detailed trace output will appear here during execution.")


def _render_result_actions(
    *,
    markdown: str,
    title: str,
    prompt: str | None,
    result: BlogRunResult | None = None,
    artifact_store: ArtifactStore | None = None,
    show_open_detail: bool,
    third_action: str,
    action_key_prefix: str = "result-actions",
) -> None:
    markdown_filename = f"{slugify_title(title)}.md"
    action_columns = st.columns(3)

    action_columns[0].download_button(
        "Download markdown",
        data=markdown.encode("utf-8"),
        file_name=markdown_filename,
        mime="text/markdown",
        use_container_width=True,
        disabled=not markdown.strip(),
        key=f"{action_key_prefix}-download-markdown",
    )

    if third_action == "bundle" and result and artifact_store:
        try:
            bundle = artifact_store.build_bundle(result)
            action_columns[1].download_button(
                "Download bundle",
                data=bundle,
                file_name=f"{slugify_title(title)}_bundle.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"{action_key_prefix}-download-bundle",
            )
        except ArtifactStoreError as exc:
            action_columns[1].info(f"Bundle unavailable: {exc}")
    elif show_open_detail:
        run_id = session.get_result_run_id(result) if result else None
        if action_columns[1].button(
            "Open Run Detail",
            disabled=not run_id,
            use_container_width=True,
            key=f"{action_key_prefix}-open-detail",
        ) and run_id:
            _open_run_detail(run_id)
    else:
        action_columns[1].button(
            "Open Run Detail",
            disabled=True,
            use_container_width=True,
            key=f"{action_key_prefix}-open-detail-disabled",
        )

    if action_columns[2].button(
        "Reuse prompt",
        disabled=not prompt,
        use_container_width=True,
        key=f"{action_key_prefix}-reuse-prompt",
    ):
        if session.get_active_page() == session.PAGE_HOME:
            session.set_blog_prompt(prompt or "")
            st.rerun()
        else:
            _open_home_with_prompt(prompt or "")


def _render_blog_tab(article_markdown: str, result: BlogRunResult) -> None:
    if not article_markdown.strip():
        render_empty_state(
            "The draft is still taking shape.",
            "The article body will render here once markdown is available.",
            ghost_steps=["Draft", "References", "Finalize"],
        )
        return

    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)

    base_dir = result.artifacts.base_dir if result.artifacts else Path(".")
    with st.container(border=True):
        render_markdown_with_local_images(article_markdown, base_dir)


def _render_outline_tab(plan_dict: dict[str, object] | None, *, fallback_title: str) -> None:
    if not plan_dict:
        render_empty_state(
            "The article outline will appear here as soon as planning finishes.",
            "Audience, tone, structure, and section goals will land here before the full draft is finalized.",
            ghost_steps=["Audience", "Structure", "Sections"],
        )
        return

    render_key_value_grid(
        [
            ("Title", str(plan_dict.get("blog_title") or fallback_title)),
            ("Audience", str(plan_dict.get("audience") or "Unknown")),
            ("Tone", str(plan_dict.get("tone") or "Unknown")),
            ("Blog kind", str(plan_dict.get("blog_kind") or "Unknown")),
        ],
        columns=2,
    )
    constraints = [str(item) for item in plan_dict.get("constraints") or [] if str(item).strip()]
    if constraints:
        render_badge_row([(constraint, "neutral") for constraint in constraints[:6]])

    for task in sorted(plan_dict.get("tasks") or [], key=lambda item: item.get("id") or 0):
        with st.container(border=True):
            st.markdown(f"**{task.get('id', '?')}. {task.get('title', 'Untitled section')}**")
            goal = str(task.get("goal") or "").strip()
            if goal:
                st.caption(goal)
            for bullet in task.get("bullets") or []:
                st.write(f"- {bullet}")
            meta_parts = []
            if task.get("target_words"):
                meta_parts.append(f"{task.get('target_words')} target words")
            if task.get("requires_research"):
                meta_parts.append("Uses research")
            if task.get("requires_citations"):
                meta_parts.append("Needs citations")
            if meta_parts:
                st.caption(" • ".join(meta_parts))


def _render_sources_tab(evidence: Sequence[EvidenceItem], queries: Sequence[str]) -> None:
    if queries:
        with st.expander("Research queries", expanded=False):
            for query in queries:
                st.write(query)

    if not evidence:
        render_empty_state(
            "Sources will collect here as research completes.",
            "The agent will keep source data out of the article body and surface it here instead.",
            ghost_steps=["Queries", "Evidence", "References"],
        )
        return

    for index, item in enumerate(evidence, start=1):
        with st.container(border=True):
            title = item.title.strip() if item.title else item.url
            st.markdown(f"{index}. [{title}]({item.url})")
            meta_parts = [part for part in [item.source, item.published_at] if part]
            if meta_parts:
                st.caption(" • ".join(meta_parts))
            if item.snippet:
                st.write(item.snippet)


def _render_images_tab(result: BlogRunResult, artifact_store: ArtifactStore) -> None:
    if result.image_specs:
        with st.expander("Image plan", expanded=False):
            specs_frame = pd.DataFrame(
                [
                    {
                        "filename": item.filename,
                        "size": item.size,
                        "quality": item.quality,
                        "caption": item.caption,
                    }
                    for item in result.image_specs
                ]
            )
            st.dataframe(specs_frame, hide_index=True)

    images_dir = result.artifacts.images_dir if result.artifacts else None
    if not images_dir or not images_dir.exists():
        message = "The run is text-only, so no visual assets were requested."
        if result.request and result.request.enable_images:
            message = "Visual assets will appear here when image generation succeeds."
        render_empty_state("Images stay attached to the run here.", message, ghost_steps=["Plan", "Generate", "Review"])
        return

    image_files = sorted(path for path in images_dir.iterdir() if path.is_file())
    if not image_files:
        render_empty_state("Images stay attached to the run here.", "The images directory was created, but no files were found.")
        return

    gallery_columns = st.columns(2)
    for index, file_path in enumerate(image_files):
        with gallery_columns[index % 2]:
            st.image(str(file_path), caption=file_path.name)

    bundle = artifact_store.build_images_bundle(result)
    if bundle:
        st.download_button("Download images", data=bundle, file_name="images.zip", mime="application/zip")


def _render_trace_tab(result: BlogRunResult) -> None:
    summary = summarize_result(result)
    render_metric_cards(
        [
            MetricCardData("Estimated cost", format_usd(summary["estimated_cost"])),
            MetricCardData("Tokens", format_int(summary["total_tokens"])),
            MetricCardData("Search calls", format_int(summary["search_calls"])),
            MetricCardData("Images", format_int(summary["images_generated"])),
        ],
        columns=4,
    )

    if result.cost_summary:
        with st.expander("Cost breakdown", expanded=False):
            provider_rows = [
                {"provider": provider, "estimated_cost_usd": amount}
                for provider, amount in result.cost_summary.by_provider.items()
            ]
            if provider_rows:
                st.dataframe(pd.DataFrame(provider_rows), hide_index=True)
            if result.cost_summary.line_items:
                st.dataframe(
                    pd.DataFrame([item.model_dump(mode="json") for item in result.cost_summary.line_items]),
                    hide_index=True,
                )

    usage_rows = _usage_rows(result)
    with st.expander("Usage records", expanded=False):
        if usage_rows:
            st.dataframe(pd.DataFrame(usage_rows), hide_index=True)
        else:
            st.caption("No structured usage records were saved for this run.")

    with st.expander("Execution log", expanded=False):
        lines = [_format_log_line(event) for event in result.events[-160:]] if result.events else session.get_logs()[-160:]
        render_log_panel(lines)

    with st.expander("Raw debug payload", expanded=False):
        st.json(result.model_dump(mode="json"))


def _render_live_result_workspace(
    *,
    request: BlogRequest,
    current_node: str | None,
    progress_summary: dict[str, object],
    warnings: Sequence[str],
    usage_rows: Sequence[dict[str, object]],
    log_lines: Sequence[str],
) -> None:
    outline = progress_summary.get("outline") if isinstance(progress_summary.get("outline"), dict) else None
    evidence = _coerce_preview_evidence(progress_summary.get("evidence_preview") or [])
    draft_preview = str(progress_summary.get("draft_preview") or "")
    article_markdown = build_renderable_blog_markdown(draft_preview, evidence) if draft_preview.strip() else ""
    blog_title = (
        str(outline.get("blog_title")) if outline and outline.get("blog_title") else request.topic.strip()
    ) or "Draft in progress"
    tasks = int(progress_summary.get("tasks") or 0)
    sections_done = min(int(progress_summary.get("sections_done") or 0), tasks) if tasks else 0

    st.caption("GENERATED RESULT")
    st.subheader("Latest draft")
    st.markdown(f"**{blog_title}**")
    status_parts = [_workflow_status_label(current_node, is_complete=False)]
    if tasks:
        status_parts.append(f"{sections_done} of {tasks} sections drafted")
    st.caption(" • ".join(status_parts))
    st.caption("Downloads and reuse actions become available once the run completes.")

    tabs = st.tabs(["Blog", "Outline", "Sources", "Images", "Trace"])
    with tabs[0]:
        if article_markdown.strip():
            with st.container(border=True):
                render_markdown_with_local_images(article_markdown, Path("."))
        elif outline:
            render_empty_state(
                "The outline is ready. The draft will update here as sections complete.",
                "The article pane refreshes in place while the agent writes.",
                ghost_steps=["Outline", "Section drafts", "Finalize"],
            )
        else:
            render_empty_state(
                "The draft will start appearing here once writing begins.",
                "Research and planning happen first, then the article updates in place.",
                ghost_steps=["Research", "Plan", "Draft"],
            )

    with tabs[1]:
        _render_outline_tab(outline, fallback_title=blog_title)

    with tabs[2]:
        _render_sources_tab(evidence, [str(query) for query in progress_summary.get("queries") or []])

    with tabs[3]:
        image_specs_preview = progress_summary.get("image_specs_preview") or []
        if image_specs_preview:
            st.dataframe(pd.DataFrame(image_specs_preview), hide_index=True)
        elif request.enable_images:
            render_empty_state(
                "Visual assets will appear here once the draft is stable.",
                "Image planning usually happens after the article structure is locked.",
                ghost_steps=["Plan", "Generate", "Review"],
            )
        else:
            render_empty_state("This run is text-only.", "Enable images in the sidebar to request visual assets.")

    with tabs[4]:
        render_badge_row(
            [
                (f"Sources {int(progress_summary.get('evidence_count') or 0)}", "neutral"),
                (f"Warnings {len(warnings)}", "neutral"),
                (f"Usage records {len(usage_rows)}", "neutral"),
            ]
        )
        if warnings:
            for warning in warnings[-2:]:
                st.warning(warning)
        with st.expander("Live trace", expanded=False):
            if usage_rows:
                st.dataframe(pd.DataFrame(usage_rows[-10:]), hide_index=True)
            render_log_panel(log_lines[-120:], empty_message="Trace events will appear here during execution.")


def _run_generation(
    request: BlogRequest,
    service,
    *,
    workflow_container,
    output_container,
    artifact_store: ArtifactStore,
) -> BlogRunResult | None:
    logs_for_run: list[str] = []
    warnings: list[str] = []
    usage_rows: list[dict[str, object]] = []
    result: BlogRunResult | None = None
    progress_summary: dict[str, object] = {}
    current_node: str | None = None
    seen_nodes: list[str] = []
    started_at = monotonic()

    workflow_placeholder = workflow_container.empty()
    output_placeholder = output_container.empty()

    try:
        for event in service.stream(request):
            logs_for_run.append(_format_log_line(event))
            if event.kind == "node":
                current_node = str(event.payload.get("name", ""))
                if current_node and current_node not in seen_nodes:
                    seen_nodes.append(current_node)
            elif event.kind == "progress":
                progress_summary = dict(event.payload)
            elif event.kind == "usage":
                usage_rows.append(event.payload.get("usage", {}))
            elif event.kind == "warning":
                warnings.append(event.message)
            elif event.kind == "result":
                result = BlogRunResult.model_validate(event.payload["result"])
                session.set_last_result(result)

            elapsed = monotonic() - started_at
            with workflow_placeholder.container():
                _render_workflow_panel(
                    current_node=current_node,
                    seen_nodes=seen_nodes,
                    progress_summary=progress_summary,
                    enable_research=request.enable_research,
                    enable_images=request.enable_images,
                    warnings=warnings,
                    log_lines=logs_for_run,
                    is_complete=result is not None,
                    elapsed_seconds=elapsed,
                )

            with output_placeholder.container():
                st.divider()
                if result:
                    _render_result_workspace(result, artifact_store, show_open_detail=True)
                else:
                    _render_live_result_workspace(
                        request=request,
                        current_node=current_node,
                        progress_summary=progress_summary,
                        warnings=warnings,
                        usage_rows=usage_rows,
                        log_lines=logs_for_run,
                    )
    except WorkflowExecutionError as exc:
        with workflow_placeholder.container():
            _render_workflow_panel(
                current_node=current_node,
                seen_nodes=seen_nodes,
                progress_summary=progress_summary,
                enable_research=request.enable_research,
                enable_images=request.enable_images,
                warnings=[*warnings, str(exc)],
                log_lines=logs_for_run,
                is_complete=False,
                elapsed_seconds=monotonic() - started_at,
            )
            st.error(str(exc))
        return None

    if logs_for_run:
        session.append_logs(logs_for_run)

    if result:
        return result

    with workflow_placeholder.container():
        _render_workflow_panel(
            current_node=current_node,
            seen_nodes=seen_nodes,
            progress_summary=progress_summary,
            enable_research=request.enable_research,
            enable_images=request.enable_images,
            warnings=warnings,
            log_lines=logs_for_run,
            is_complete=False,
            elapsed_seconds=monotonic() - started_at,
        )
        st.error("The workflow completed without returning a final result.")
    return None


def _extract_execution_state(events: Sequence[RunEvent]) -> tuple[str | None, list[str], dict[str, object]]:
    current_node: str | None = None
    seen_nodes: list[str] = []
    progress_summary: dict[str, object] = {}

    for event in events:
        if event.kind == "node":
            current_node = str(event.payload.get("name", ""))
            if current_node and current_node not in seen_nodes:
                seen_nodes.append(current_node)
        elif event.kind == "progress":
            progress_summary = dict(event.payload)
    return current_node, seen_nodes, progress_summary


def _coerce_preview_evidence(items: Sequence[object]) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    for item in items:
        if isinstance(item, EvidenceItem):
            evidence.append(item)
            continue
        if isinstance(item, dict):
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or url).strip()
            if not url or not title:
                continue
            evidence.append(
                EvidenceItem(
                    title=title,
                    url=url,
                    published_at=str(item.get("published_at") or "").strip() or None,
                    snippet=str(item.get("snippet") or "").strip() or None,
                    source=str(item.get("source") or "").strip() or None,
                )
            )
    return evidence


def _result_updated_timestamp(result: BlogRunResult) -> str | None:
    artifacts = result.artifacts
    if not artifacts or not artifacts.markdown_path.exists():
        return None
    try:
        return datetime.fromtimestamp(artifacts.markdown_path.stat().st_mtime, UTC).isoformat()
    except OSError:
        return None


def _workflow_status_label(current_node: str | None, *, is_complete: bool) -> str:
    if is_complete:
        return "Finalized article"
    if current_node == "research":
        return "Researching sources"
    if current_node == "orchestrator":
        return "Planning outline"
    if current_node == "worker":
        return "Drafting article"
    if current_node == "merge_content":
        return "Finalizing article"
    if current_node == "decide_images":
        return "Planning visual assets"
    if current_node == "generate_and_place_images":
        return "Generating images"
    if current_node == "router":
        return "Assessing request"
    return "Starting run"


def _build_stage_tracker(
    current_node: str | None,
    seen_nodes: Sequence[str],
    progress_summary: dict[str, object],
    *,
    enable_research: bool,
    enable_images: bool,
    is_complete: bool,
) -> list[tuple[str, str]]:
    current_stage_index = _high_level_stage_index(current_node)
    stages = ["Research", "Plan", "Write", "Finalize"]
    items: list[tuple[str, str]] = []

    for index, label in enumerate(stages):
        if label == "Research" and not bool(progress_summary.get("needs_research", enable_research)):
            state = "Skipped"
        elif is_complete:
            state = "Done"
        elif current_stage_index > index:
            state = "Done"
        elif current_stage_index == index:
            state = "Live"
        else:
            state = "Queued"

        if label == "Finalize" and not enable_images and current_node == "decide_images":
            state = "Live"
        items.append((label, state))

    if current_stage_index < 0 and not seen_nodes and not is_complete:
        return [(label, "Ready") for label in stages]
    return items


def _high_level_stage_index(current_node: str | None) -> int:
    stage_map = {
        "router": 0,
        "research": 0,
        "orchestrator": 1,
        "worker": 2,
        "merge_content": 3,
        "decide_images": 3,
        "generate_and_place_images": 3,
    }
    return stage_map.get(current_node, -1)


def _build_execution_timeline(
    current_node: str | None,
    seen_nodes: Sequence[str],
    progress_summary: dict[str, object],
    *,
    enable_research: bool,
    enable_images: bool,
    is_complete: bool,
) -> list[TimelineStep]:
    steps: list[TimelineStep] = []
    current_index = _STAGE_ORDER.index(current_node) if current_node in _STAGE_ORDER else -1
    needs_research = bool(progress_summary.get("needs_research", enable_research))
    tasks = int(progress_summary.get("tasks") or 0)
    sections_done = int(progress_summary.get("sections_done") or 0)
    evidence_count = int(progress_summary.get("evidence_count") or 0)
    queries = [str(query) for query in progress_summary.get("queries") or []]
    outline = progress_summary.get("outline") if isinstance(progress_summary.get("outline"), dict) else None
    outline_tasks = outline.get("tasks") if outline else []

    if not needs_research:
        steps.append(TimelineStep(title="Skipping external research for this run", meta="Closed-book drafting", state="skipped"))
    else:
        research_state = "active" if current_node in {"router", "research"} else "completed" if current_index > 1 or "research" in seen_nodes else "pending"
        evidence_state = "active" if current_node == "research" else "completed" if evidence_count or current_index > 1 else "pending"
        steps.append(
            TimelineStep(
                title="Searching for relevant sources",
                meta=f"{len(queries)} queries queued" if queries else "Preparing research plan",
                state=research_state,
            )
        )
        steps.append(
            TimelineStep(
                title="Extracting evidence from sources",
                meta=f"{evidence_count} sources captured" if evidence_count else "Waiting for source results",
                state=evidence_state,
            )
        )

    planning_state = "active" if current_node == "orchestrator" else "completed" if tasks or current_index > 2 or "orchestrator" in seen_nodes else "pending"
    steps.append(
        TimelineStep(
            title="Planning article structure",
            meta=f"{tasks} sections planned" if tasks else "Waiting for outline",
            state=planning_state,
        )
    )

    drafting_state = "active" if current_node == "worker" else "completed" if (tasks and sections_done >= tasks and current_node != "worker") or current_index > 3 else "pending"
    steps.append(
        TimelineStep(
            title="Drafting article sections",
            meta=f"{min(sections_done, tasks)} of {tasks} sections complete" if tasks else "Waiting for section plan",
            state=drafting_state,
        )
    )
    if current_node == "worker" and tasks:
        next_index = min(sections_done, max(tasks - 1, 0))
        next_title = None
        if isinstance(outline_tasks, list) and next_index < len(outline_tasks):
            next_title = str(outline_tasks[next_index].get("title") or "").strip() or None
        steps.append(
            TimelineStep(
                title=f"Drafting next section: {next_title or f'Section {next_index + 1}'}",
                meta="Current writing task",
                state="active",
            )
        )

    if enable_images:
        image_specs = progress_summary.get("image_specs_preview") or []
        image_state = "active" if current_node in {"decide_images", "generate_and_place_images"} else "completed" if current_index > 5 or bool(image_specs) and current_index > 5 else "pending"
        steps.append(
            TimelineStep(
                title="Preparing visual assets",
                meta=f"{len(image_specs)} image briefs ready" if image_specs else "Visual planning will follow the draft",
                state=image_state,
            )
        )

    final_state = (
        "completed"
        if is_complete
        else "active"
        if current_node in {"merge_content", "generate_and_place_images"} or (not enable_images and current_node == "decide_images")
        else "pending"
    )
    steps.append(
        TimelineStep(
            title="Preparing final output",
            meta="References, assets, and trace are being attached" if final_state == "active" else "Waiting for final assembly",
            state=final_state,
        )
    )
    return steps


def _estimate_execution_progress(
    current_node: str | None,
    progress_summary: dict[str, object],
    *,
    is_complete: bool,
) -> float:
    if is_complete:
        return 1.0

    tasks = max(int(progress_summary.get("tasks") or 0), 1)
    sections_done = min(int(progress_summary.get("sections_done") or 0), tasks)
    stage_progress = {
        None: 0.05,
        "router": 0.12,
        "research": 0.24,
        "orchestrator": 0.38,
        "worker": 0.42 + (0.32 * (sections_done / tasks)),
        "merge_content": 0.82,
        "decide_images": 0.9,
        "generate_and_place_images": 0.96,
    }
    return min(stage_progress.get(current_node, 0.08), 0.99)


def _execution_progress_label(
    current_node: str | None,
    progress_summary: dict[str, object],
    *,
    is_complete: bool,
) -> str:
    if is_complete:
        return "Run complete"
    tasks = int(progress_summary.get("tasks") or 0)
    sections_done = int(progress_summary.get("sections_done") or 0)
    if current_node == "worker" and tasks:
        return f"Drafting article ({min(sections_done, tasks)}/{tasks} sections)"
    if current_node == "research":
        return "Researching sources"
    if current_node == "orchestrator":
        return "Planning outline"
    if current_node == "merge_content":
        return "Finalizing article"
    if current_node == "decide_images":
        return "Planning images"
    if current_node == "generate_and_place_images":
        return "Generating images"
    return _STAGE_LABELS.get(current_node or "", "Initializing workflow")


def _render_saved_run_row(saved_blog: SavedBlog, _artifact_store: ArtifactStore, *, card_prefix: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{saved_blog.title}**")
        st.caption(saved_blog.request_topic or saved_blog.title)
        meta_parts = [format_timestamp(saved_blog.created_at), saved_blog.status]
        if saved_blog.provider_mix:
            meta_parts.append(", ".join(saved_blog.provider_mix))
        if saved_blog.cost_summary:
            meta_parts.append(format_usd(saved_blog.cost_summary.total_estimated_cost_usd))
        st.caption(" • ".join(meta_parts))
        if saved_blog.cost_summary:
            detail_parts = []
            if saved_blog.cost_summary.total_tokens:
                detail_parts.append(f"{format_int(saved_blog.cost_summary.total_tokens)} tokens")
            if saved_blog.cost_summary.search_calls:
                detail_parts.append(f"{saved_blog.cost_summary.search_calls} searches")
            if saved_blog.cost_summary.images_generated:
                detail_parts.append(f"{saved_blog.cost_summary.images_generated} images")
            if detail_parts:
                st.caption(" • ".join(detail_parts))

        action_columns = st.columns(2)
        if action_columns[0].button("Open", key=f"{card_prefix}-open-{saved_blog.run_id}"):
            _open_run_detail(saved_blog.run_id)
        if action_columns[1].button("Reuse prompt", key=f"{card_prefix}-prompt-{saved_blog.run_id}"):
            _open_home_with_prompt(saved_blog.request_topic or saved_blog.title)


def _provider_rows(statuses: Iterable[ProviderStatus]) -> list[dict[str, object]]:
    return [
        {
            "provider": status.provider,
            "state": status.state,
            "ready": status.ready,
            "credential_source": status.credential_source,
            "model": status.model,
            "message": status.message,
        }
        for status in statuses
    ]


def _usage_rows(result: BlogRunResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in result.usage_records:
        if hasattr(record, "model_dump"):
            rows.append(record.model_dump(mode="json"))
        else:
            rows.append(dict(record))
    return rows


def _resolve_selected_blog(saved_runs: Sequence[SavedBlog]) -> tuple[SavedBlog | None, bool]:
    if not saved_runs:
        return None, False

    selected_run_id = session.get_selected_run_id()
    if not selected_run_id:
        return saved_runs[0], False

    for saved_blog in saved_runs:
        if saved_blog.run_id == selected_run_id:
            return saved_blog, False
    return saved_runs[0], True


def _resolve_selected_result(artifact_store: ArtifactStore, selected_blog: SavedBlog | None) -> BlogRunResult | None:
    last_result = session.get_last_result()
    selected_run_id = selected_blog.run_id if selected_blog else session.get_selected_run_id()

    if last_result and selected_run_id and session.get_result_run_id(last_result) == selected_run_id:
        return last_result
    if last_result and not selected_blog:
        return last_result
    if not selected_blog:
        return None

    try:
        loaded_result = artifact_store.read_run(selected_blog)
    except ArtifactStoreError as exc:
        st.error(str(exc))
        return None

    session.set_last_result(loaded_result)
    session.set_selected_run_id(selected_blog.run_id)
    return loaded_result


def _open_home_with_prompt(prompt: str) -> None:
    session.set_blog_prompt(prompt)
    if not session.navigate_to(session.PAGE_HOME):
        st.error("Home is unavailable. Reload the app and try again.")


def _open_run_detail(run_id: str) -> None:
    session.set_selected_run_id(run_id)
    if not session.navigate_to(session.PAGE_RUN_DETAIL):
        st.error("Run Detail is unavailable. Reload the app and try again.")


def _sync_settings_widgets(runtime_config: ResolvedRuntimeConfig, overrides: SessionRuntimeConfig) -> None:
    st.session_state.setdefault(session.SETTINGS_OPENAI_KEY_KEY, overrides.openai_api_key or "")
    st.session_state.setdefault(session.SETTINGS_TAVILY_KEY_KEY, overrides.tavily_api_key or "")
    st.session_state.setdefault(session.SETTINGS_GOOGLE_KEY_KEY, overrides.google_api_key or "")
    st.session_state.setdefault(session.SETTINGS_LANGSMITH_API_KEY, overrides.langsmith_api_key or "")
    st.session_state.setdefault(session.SETTINGS_OPENAI_MODEL_KEY, overrides.openai_model or runtime_config.openai_model)
    st.session_state.setdefault(
        session.SETTINGS_GOOGLE_IMAGE_MODEL_KEY,
        overrides.google_image_model or runtime_config.google_image_model,
    )
    st.session_state.setdefault(
        session.SETTINGS_ENABLE_RESEARCH_KEY,
        runtime_config.default_enable_research if overrides.default_enable_research is None else overrides.default_enable_research,
    )
    st.session_state.setdefault(
        session.SETTINGS_ENABLE_IMAGES_KEY,
        runtime_config.default_enable_images if overrides.default_enable_images is None else overrides.default_enable_images,
    )
    st.session_state.setdefault(
        session.SETTINGS_LANGSMITH_TRACING_KEY,
        runtime_config.langsmith_tracing if overrides.langsmith_tracing is None else overrides.langsmith_tracing,
    )
    st.session_state.setdefault(
        session.SETTINGS_LANGSMITH_PROJECT_KEY,
        overrides.langsmith_project or runtime_config.langsmith_project or "",
    )
    st.session_state.setdefault(
        session.SETTINGS_PRICING_LABEL_KEY,
        overrides.pricing.label if overrides.pricing else runtime_config.pricing.label,
    )
    current_openai_model = st.session_state[session.SETTINGS_OPENAI_MODEL_KEY]
    current_image_model = st.session_state[session.SETTINGS_GOOGLE_IMAGE_MODEL_KEY]
    pricing = overrides.pricing or runtime_config.pricing
    st.session_state.setdefault(
        session.SETTINGS_OPENAI_INPUT_PRICE_KEY,
        pricing.openai_models.get(current_openai_model, pricing.openai_models.get(runtime_config.openai_model)).input_per_1m_tokens_usd,
    )
    st.session_state.setdefault(
        session.SETTINGS_OPENAI_OUTPUT_PRICE_KEY,
        pricing.openai_models.get(current_openai_model, pricing.openai_models.get(runtime_config.openai_model)).output_per_1m_tokens_usd,
    )
    st.session_state.setdefault(session.SETTINGS_TAVILY_QUERY_PRICE_KEY, pricing.tavily_search.per_query_usd)
    st.session_state.setdefault(
        session.SETTINGS_GOOGLE_IMAGE_PRICE_KEY,
        pricing.google_image_models.get(
            current_image_model,
            pricing.google_image_models.get(runtime_config.google_image_model),
        ).per_image_usd,
    )


def _reset_settings_widgets(base_settings: AppSettings) -> None:
    st.session_state[session.SETTINGS_OPENAI_KEY_KEY] = ""
    st.session_state[session.SETTINGS_TAVILY_KEY_KEY] = ""
    st.session_state[session.SETTINGS_GOOGLE_KEY_KEY] = ""
    st.session_state[session.SETTINGS_LANGSMITH_API_KEY] = ""
    st.session_state[session.SETTINGS_OPENAI_MODEL_KEY] = base_settings.openai_model
    st.session_state[session.SETTINGS_GOOGLE_IMAGE_MODEL_KEY] = base_settings.google_image_model
    st.session_state[session.SETTINGS_ENABLE_RESEARCH_KEY] = base_settings.default_enable_research
    st.session_state[session.SETTINGS_ENABLE_IMAGES_KEY] = base_settings.default_enable_images
    st.session_state[session.SETTINGS_LANGSMITH_TRACING_KEY] = base_settings.langsmith_tracing
    st.session_state[session.SETTINGS_LANGSMITH_PROJECT_KEY] = base_settings.langsmith_project or ""
    st.session_state[session.SETTINGS_PRICING_LABEL_KEY] = base_settings.pricing.label
    st.session_state[session.SETTINGS_OPENAI_INPUT_PRICE_KEY] = (
        base_settings.pricing.openai_models[base_settings.openai_model].input_per_1m_tokens_usd
    )
    st.session_state[session.SETTINGS_OPENAI_OUTPUT_PRICE_KEY] = (
        base_settings.pricing.openai_models[base_settings.openai_model].output_per_1m_tokens_usd
    )
    st.session_state[session.SETTINGS_TAVILY_QUERY_PRICE_KEY] = base_settings.pricing.tavily_search.per_query_usd
    st.session_state[session.SETTINGS_GOOGLE_IMAGE_PRICE_KEY] = (
        base_settings.pricing.google_image_models[base_settings.google_image_model].per_image_usd
    )


def _build_runtime_overrides_from_widgets() -> SessionRuntimeConfig:
    openai_model = st.session_state[session.SETTINGS_OPENAI_MODEL_KEY].strip() or "gpt-4.1-mini"
    image_model = st.session_state[session.SETTINGS_GOOGLE_IMAGE_MODEL_KEY].strip() or "gemini-2.5-flash-image"
    pricing = PricingConfig(
        label=st.session_state[session.SETTINGS_PRICING_LABEL_KEY].strip() or "Session overrides",
        openai_models={
            openai_model: {
                "input_per_1m_tokens_usd": float(st.session_state[session.SETTINGS_OPENAI_INPUT_PRICE_KEY]),
                "output_per_1m_tokens_usd": float(st.session_state[session.SETTINGS_OPENAI_OUTPUT_PRICE_KEY]),
            }
        },
        tavily_search={"per_query_usd": float(st.session_state[session.SETTINGS_TAVILY_QUERY_PRICE_KEY])},
        google_image_models={
            image_model: {"per_image_usd": float(st.session_state[session.SETTINGS_GOOGLE_IMAGE_PRICE_KEY])}
        },
    )
    return SessionRuntimeConfig(
        openai_api_key=st.session_state[session.SETTINGS_OPENAI_KEY_KEY].strip() or None,
        tavily_api_key=st.session_state[session.SETTINGS_TAVILY_KEY_KEY].strip() or None,
        google_api_key=st.session_state[session.SETTINGS_GOOGLE_KEY_KEY].strip() or None,
        langsmith_api_key=st.session_state[session.SETTINGS_LANGSMITH_API_KEY].strip() or None,
        openai_model=openai_model,
        google_image_model=image_model,
        default_enable_research=bool(st.session_state[session.SETTINGS_ENABLE_RESEARCH_KEY]),
        default_enable_images=bool(st.session_state[session.SETTINGS_ENABLE_IMAGES_KEY]),
        langsmith_tracing=bool(st.session_state[session.SETTINGS_LANGSMITH_TRACING_KEY]),
        langsmith_project=st.session_state[session.SETTINGS_LANGSMITH_PROJECT_KEY].strip() or None,
        pricing=PricingConfig.model_validate(pricing),
    )


def _format_log_line(event: RunEvent) -> str:
    if event.kind == "node":
        return f"[node] {event.payload.get('name', 'unknown')}"
    if event.kind == "warning":
        return f"[warning] {event.message}"
    if event.kind == "error":
        return f"[error] {event.message}"
    if event.kind == "progress":
        return f"[progress] {event.payload}"
    if event.kind == "usage":
        return f"[usage] {event.payload}"
    if event.kind == "result":
        return "[result] Final result available."
    return f"[{event.kind}] {event.message}"
