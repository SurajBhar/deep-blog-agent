"""Shared visual helpers for the Streamlit UI."""

from __future__ import annotations

from typing import Iterable, Sequence

import streamlit as st


def inject_global_styles() -> None:
    """Apply a minimal global theme once per app run."""

    st.markdown(
        """
        <style>
        .stApp {
            background: #f8f9fb;
            color: #111827;
        }

        .block-container {
            max-width: 1080px;
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid rgba(17, 24, 39, 0.08);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 0.75rem;
            padding-bottom: 1rem;
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.35rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.94);
        }

        .stTextArea textarea,
        .stTextInput input,
        .stDateInput input,
        .stNumberInput input,
        div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            border-radius: 12px !important;
        }

        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] > button {
            border-radius: 999px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding-inline: 0.75rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    title: str,
    subtitle: str,
    *,
    eyebrow: str | None = None,
    chips: Iterable[str] | None = None,
) -> None:
    """Render a minimal page header."""

    if eyebrow:
        st.caption(eyebrow)
    st.title(title)
    st.caption(subtitle)
    if chips:
        visible = [chip for chip in chips if chip]
        if visible:
            st.caption(" • ".join(visible))
    st.divider()


def render_section_intro(label: str, title: str, description: str | None = None) -> None:
    """Render a consistent section heading."""

    if label:
        st.caption(label.upper())
    st.subheader(title)
    if description:
        st.caption(description)


def render_card(title: str, copy: str, *, eyebrow: str | None = None, footer: str | None = None) -> None:
    """Render a simple bordered container."""

    with st.container(border=True):
        if eyebrow:
            st.caption(eyebrow)
        st.markdown(f"**{title}**")
        st.write(copy)
        if footer:
            st.caption(footer)


def render_sidebar_brand(title: str, copy: str, *, bullets: Sequence[str] | None = None) -> None:
    """Render the sidebar brand block."""

    st.markdown(f"### {title}")
    st.caption(copy)
    if bullets:
        for item in bullets:
            st.caption(f"• {item}")
    st.divider()


def render_sidebar_snapshot(title: str, items: Sequence[tuple[str, str]]) -> None:
    """Render a compact sidebar snapshot block."""

    st.markdown(f"**{title}**")
    for label, value in items:
        st.caption(f"{label}: {value}")
    st.divider()


def render_sidebar_about(*, repository_url: str) -> None:
    """Render a compact about block near the bottom of the sidebar."""

    with st.expander("About", expanded=False):
        st.caption("Developed by Suraj Bhardwaj")
        st.caption("MIT License")
        st.markdown(f"[GitHub repo]({repository_url})")


def render_status_strip(items: list[tuple[str, str]]) -> None:
    """Render a compact status row."""

    columns = st.columns(len(items) or 1)
    for column, (label, value) in zip(columns, items):
        with column:
            st.caption(label)
            st.write(value)
