"""Streamlit multipage application entrypoint."""

from __future__ import annotations

import streamlit as st

from deep_blog_agent.blog_writer.service import build_default_blog_generation_service
from deep_blog_agent.core.runtime import resolve_runtime
from deep_blog_agent.core.settings import AppSettings
from . import session
from .pages import (
    UIContext,
    render_finops_page,
    render_home_page,
    render_prompt_examples_page,
    render_run_detail_page,
    render_run_history_page,
    render_settings_page,
)
from .theme import inject_global_styles, render_sidebar_about


def main(service=None) -> None:
    runtime_service = service or build_default_blog_generation_service()
    base_settings = runtime_service.settings if service else AppSettings.load()

    st.set_page_config(page_title="Deep Blog Agent", layout="wide")
    inject_global_styles()
    session.ensure_defaults()

    resolved_runtime = resolve_runtime(base_settings, session.get_runtime_overrides())
    session.ensure_run_option_defaults(
        default_enable_research=resolved_runtime.config.default_enable_research,
        default_enable_images=resolved_runtime.config.default_enable_images,
    )
    context = UIContext(
        base_settings=base_settings,
        resolved_runtime=resolved_runtime,
        artifact_store=runtime_service.artifact_store,
    )

    def home_page() -> None:
        render_home_page(context)

    def prompt_examples_page() -> None:
        render_prompt_examples_page(context)

    def run_history_page() -> None:
        render_run_history_page(context)

    def run_detail_page() -> None:
        render_run_detail_page(context)

    def settings_page() -> None:
        render_settings_page(context)

    def finops_page() -> None:
        render_finops_page(context)

    pages = {
        session.PAGE_HOME: st.Page(home_page, title="Home", default=True),
        session.PAGE_PROMPT_EXAMPLES: st.Page(prompt_examples_page, title="Prompt Examples"),
        session.PAGE_RUN_HISTORY: st.Page(run_history_page, title="Run History"),
        session.PAGE_RUN_DETAIL: st.Page(run_detail_page, title="Run Detail"),
        session.PAGE_SETTINGS: st.Page(settings_page, title="Settings"),
        session.PAGE_FINOPS: st.Page(finops_page, title="FinOps"),
    }
    session.register_pages(pages)
    navigation = st.navigation(list(pages.values()))
    session.sync_active_page(navigation)

    with st.sidebar:
        st.markdown("**Run options**")
        st.date_input("As-of date", key=session.AS_OF_INPUT_KEY)
        st.toggle("Enable research", key=session.CREATE_ENABLE_RESEARCH_KEY)
        st.toggle("Enable images", key=session.CREATE_ENABLE_IMAGES_KEY)
        st.divider()

        st.markdown("**Configurations**")
        if _is_openai_missing(resolved_runtime.settings):
            st.caption("OPENAI_API_KEY is required before you can generate a draft.")
        else:
            st.caption("Use Settings to edit API keys, LangSmith options, models, and pricing.")
        if st.button("Open Settings", use_container_width=True, key="sidebar-open-settings"):
            session.navigate_to(session.PAGE_SETTINGS)
        st.divider()

    navigation.run()

    with st.sidebar:
        render_sidebar_about(repository_url="https://github.com/SurajBhar/deep-blog-agent")


def _is_openai_missing(settings: AppSettings) -> bool:
    return not bool((settings.openai_api_key or "").strip())
