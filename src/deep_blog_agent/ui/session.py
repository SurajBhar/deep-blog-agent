"""Streamlit session-state helpers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import streamlit as st

from deep_blog_agent.blog_writer.contracts import BlogRunResult, SessionRuntimeConfig

LAST_RESULT_KEY = "deep_blog_agent.last_result"
LOGS_KEY = "deep_blog_agent.logs"
BLOG_PROMPT_KEY = "deep_blog_agent.blog_prompt"
HOME_TOPIC_INPUT_KEY = BLOG_PROMPT_KEY
TOPIC_INPUT_KEY = BLOG_PROMPT_KEY
AS_OF_INPUT_KEY = "deep_blog_agent.as_of_input"
SAVED_BLOG_SELECTION_KEY = "deep_blog_agent.saved_blog_selection"
SELECTED_RUN_ID_KEY = "deep_blog_agent.selected_run_id"
ACTIVE_PAGE_KEY = "deep_blog_agent.active_page"
SESSION_RUNTIME_CONFIG_KEY = "deep_blog_agent.session_runtime_config"
CREATE_ENABLE_RESEARCH_KEY = "deep_blog_agent.create_enable_research"
CREATE_ENABLE_IMAGES_KEY = "deep_blog_agent.create_enable_images"

SETTINGS_OPENAI_KEY_KEY = "deep_blog_agent.settings.openai_api_key"
SETTINGS_TAVILY_KEY_KEY = "deep_blog_agent.settings.tavily_api_key"
SETTINGS_GOOGLE_KEY_KEY = "deep_blog_agent.settings.google_api_key"
SETTINGS_LANGSMITH_API_KEY = "deep_blog_agent.settings.langsmith_api_key"
SETTINGS_OPENAI_MODEL_KEY = "deep_blog_agent.settings.openai_model"
SETTINGS_GOOGLE_IMAGE_MODEL_KEY = "deep_blog_agent.settings.google_image_model"
SETTINGS_ENABLE_RESEARCH_KEY = "deep_blog_agent.settings.default_enable_research"
SETTINGS_ENABLE_IMAGES_KEY = "deep_blog_agent.settings.default_enable_images"
SETTINGS_LANGSMITH_TRACING_KEY = "deep_blog_agent.settings.langsmith_tracing"
SETTINGS_LANGSMITH_PROJECT_KEY = "deep_blog_agent.settings.langsmith_project"
SETTINGS_PRICING_LABEL_KEY = "deep_blog_agent.settings.pricing_label"
SETTINGS_OPENAI_INPUT_PRICE_KEY = "deep_blog_agent.settings.openai_input_price"
SETTINGS_OPENAI_OUTPUT_PRICE_KEY = "deep_blog_agent.settings.openai_output_price"
SETTINGS_TAVILY_QUERY_PRICE_KEY = "deep_blog_agent.settings.tavily_query_price"
SETTINGS_GOOGLE_IMAGE_PRICE_KEY = "deep_blog_agent.settings.google_image_price"

PAGE_HOME = "home"
PAGE_PROMPT_EXAMPLES = "prompt_examples"
PAGE_RUN_HISTORY = "run_history"
PAGE_RUN_DETAIL = "run_detail"
PAGE_SETTINGS = "settings"
PAGE_FINOPS = "finops"

_PAGE_REGISTRY: dict[str, object] = {}


def ensure_defaults() -> None:
    if LAST_RESULT_KEY not in st.session_state:
        st.session_state[LAST_RESULT_KEY] = None
    if LOGS_KEY not in st.session_state:
        st.session_state[LOGS_KEY] = []
    if BLOG_PROMPT_KEY not in st.session_state:
        st.session_state[BLOG_PROMPT_KEY] = ""
    if AS_OF_INPUT_KEY not in st.session_state:
        st.session_state[AS_OF_INPUT_KEY] = date.today()
    if SELECTED_RUN_ID_KEY not in st.session_state:
        st.session_state[SELECTED_RUN_ID_KEY] = None
    if ACTIVE_PAGE_KEY not in st.session_state:
        st.session_state[ACTIVE_PAGE_KEY] = PAGE_HOME
    if SESSION_RUNTIME_CONFIG_KEY not in st.session_state:
        st.session_state[SESSION_RUNTIME_CONFIG_KEY] = SessionRuntimeConfig()


def ensure_run_option_defaults(*, default_enable_research: bool, default_enable_images: bool) -> None:
    if CREATE_ENABLE_RESEARCH_KEY not in st.session_state:
        st.session_state[CREATE_ENABLE_RESEARCH_KEY] = default_enable_research
    if CREATE_ENABLE_IMAGES_KEY not in st.session_state:
        st.session_state[CREATE_ENABLE_IMAGES_KEY] = default_enable_images


def register_pages(pages: Mapping[str, object]) -> None:
    _PAGE_REGISTRY.clear()
    _PAGE_REGISTRY.update(dict(pages))


def sync_active_page(current_page: object) -> None:
    for page_name, page in _PAGE_REGISTRY.items():
        if page == current_page:
            st.session_state[ACTIVE_PAGE_KEY] = page_name
            return


def navigate_to(page_name: str) -> bool:
    st.session_state[ACTIVE_PAGE_KEY] = page_name
    page = _PAGE_REGISTRY.get(page_name)
    if page is None:
        return False
    st.switch_page(page)
    return True


def set_last_result(result: BlogRunResult) -> None:
    st.session_state[LAST_RESULT_KEY] = result
    run_id = get_result_run_id(result)
    if run_id:
        st.session_state[SELECTED_RUN_ID_KEY] = run_id


def get_last_result() -> BlogRunResult | None:
    return st.session_state.get(LAST_RESULT_KEY)


def append_logs(lines: list[str]) -> None:
    st.session_state[LOGS_KEY].extend(lines)


def get_logs() -> list[str]:
    return st.session_state.get(LOGS_KEY, [])


def get_home_topic_input() -> str:
    return get_blog_prompt()


def set_home_topic_input(topic: str) -> None:
    set_blog_prompt(topic)


def set_blog_prompt(topic: str) -> None:
    st.session_state[BLOG_PROMPT_KEY] = topic


def get_blog_prompt() -> str:
    return st.session_state.get(BLOG_PROMPT_KEY, "")


def queue_topic_input(topic: str) -> None:
    set_blog_prompt(topic)


def get_topic_input() -> str:
    return get_blog_prompt()


def reset_logs() -> None:
    st.session_state[LOGS_KEY] = []


def get_selected_run_id() -> str | None:
    return st.session_state.get(SELECTED_RUN_ID_KEY)


def set_selected_run_id(run_id: str | None) -> None:
    st.session_state[SELECTED_RUN_ID_KEY] = run_id


def get_active_page() -> str:
    return st.session_state.get(ACTIVE_PAGE_KEY, PAGE_HOME)


def get_runtime_overrides() -> SessionRuntimeConfig:
    value = st.session_state.get(SESSION_RUNTIME_CONFIG_KEY, SessionRuntimeConfig())
    if isinstance(value, SessionRuntimeConfig):
        return value
    return SessionRuntimeConfig.model_validate(value)


def set_runtime_overrides(config: SessionRuntimeConfig) -> None:
    st.session_state[SESSION_RUNTIME_CONFIG_KEY] = config


def clear_runtime_overrides() -> None:
    st.session_state[SESSION_RUNTIME_CONFIG_KEY] = SessionRuntimeConfig()


def get_result_run_id(result: BlogRunResult | None) -> str | None:
    if not result or not result.artifacts or not result.artifacts.run_dir:
        return None
    return result.artifacts.run_dir.name
