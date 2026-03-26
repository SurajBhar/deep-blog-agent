"""Reusable Streamlit UI components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import streamlit as st


@dataclass(frozen=True)
class MetricCardData:
    label: str
    value: str
    hint: str | None = None
    delta: str | None = None
    tone: str = "neutral"


@dataclass(frozen=True)
class TimelineStep:
    title: str
    meta: str | None = None
    state: str = "pending"


def render_metric_cards(cards: Sequence[MetricCardData], *, columns: int | None = None) -> None:
    """Render metrics using native Streamlit metric widgets."""

    if not cards:
        return

    column_count = columns or min(len(cards), 4)
    metric_columns = st.columns(column_count)
    for index, card in enumerate(cards):
        with metric_columns[index % column_count]:
            st.metric(card.label, card.value, delta=card.delta)
            if card.hint:
                st.caption(card.hint)


def render_key_value_grid(items: Sequence[tuple[str, str]], *, columns: int = 2) -> None:
    """Render key-value pairs with native Streamlit text."""

    if not items:
        return

    column_count = max(1, columns)
    grid_columns = st.columns(column_count)
    for index, (label, value) in enumerate(items):
        with grid_columns[index % column_count]:
            st.caption(label)
            st.write(value)


def render_empty_state(title: str, message: str, *, ghost_steps: Sequence[str] | None = None) -> None:
    """Render a lightweight empty state without dashboard-style placeholders."""

    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(message)
        if ghost_steps:
            st.caption(" -> ".join(step for step in ghost_steps if step))


def render_badge_row(items: Sequence[tuple[str, str]]) -> None:
    """Render lightweight inline labels."""

    labels = [label for label, _kind in items if label]
    if labels:
        st.caption(" • ".join(labels))


def render_timeline(steps: Sequence[TimelineStep], *, empty_message: str = "No workflow activity yet.") -> None:
    """Render a lightweight workflow timeline."""

    if not steps:
        render_empty_state("No activity yet", empty_message)
        return

    state_labels = {
        "completed": "Done",
        "active": "Live",
        "skipped": "Skipped",
        "pending": "Queued",
    }
    for step in steps:
        state_label = state_labels.get(step.state, step.state.title())
        st.write(f"**{state_label}**  {step.title}")
        meta = step.meta or state_label
        st.caption(meta)


def render_log_panel(lines: Sequence[str], *, empty_message: str = "No log events yet.", language: str = "text") -> None:
    """Render a log/code block."""

    st.code("\n".join(lines) if lines else empty_message, language=language)
