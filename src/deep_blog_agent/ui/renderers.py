"""UI rendering helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import streamlit as st

from deep_blog_agent.artifacts.utils import resolve_markdown_image_path
from deep_blog_agent.blog_writer.contracts import BlogRunResult, EvidenceItem, Plan

_MARKDOWN_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<caption>.+)\*$")
_CODE_FENCE_RE = re.compile(r"(?s)```.*?```")
_HEADING_RE = re.compile(r"(?m)^(#{1,6})([^\s#])")
_BULLET_RE = re.compile(r"(?m)^([ \t]*)[•◦▪]\s+")
_LIST_RE = re.compile(r"(?m)^([ \t]*)([-*+]|\d+\.)[ \t]*([^\s])")
_ORDERED_PAREN_RE = re.compile(r"(?m)^([ \t]*\d+)\)[ \t]*([^\s])")
_BLOCKQUOTE_RE = re.compile(r"(?m)^(>+)([^\s>])")
_REFERENCES_HEADING_RE = re.compile(r"(?im)^#{1,6}\s+references\s*$")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CONTROL_TRANSLATION = str.maketrans(
    {
        "\x12": "-",
        "\x13": "-",
        "\x14": "-",
        "\x15": "-",
        "\x18": "'",
        "\x19": "'",
    }
)


def normalize_markdown(markdown: str) -> str:
    normalized = (markdown or "").replace("\r\n", "\n").replace("\r", "\n").translate(_CONTROL_TRANSLATION)
    normalized = _CONTROL_CHAR_RE.sub("", normalized).strip()
    if not normalized:
        return ""

    parts: list[str] = []
    cursor = 0
    for match in _CODE_FENCE_RE.finditer(normalized):
        prose = normalized[cursor : match.start()]
        if prose:
            parts.append(_normalize_markdown_prose(prose))
        parts.append(match.group(0))
        cursor = match.end()

    tail = normalized[cursor:]
    if tail:
        parts.append(_normalize_markdown_prose(tail))

    normalized = "".join(parts) if parts else _normalize_markdown_prose(normalized)
    if normalized.count("```") % 2 == 1:
        normalized = normalized.rstrip() + "\n```"
    return normalized.rstrip() + "\n"


def build_renderable_blog_markdown(markdown: str, evidence: list[EvidenceItem]) -> str:
    normalized = normalize_markdown(markdown)
    if not normalized:
        return ""

    body = _strip_existing_references_section(normalized)
    references = _build_reference_lines(evidence)
    if not references:
        return body + "\n"
    return f"{body}\n\n## References\n\n" + "\n".join(references) + "\n"


def split_markdown_for_rendering(markdown: str) -> list[tuple[str, str]]:
    matches = list(_MARKDOWN_IMAGE_RE.finditer(markdown))
    if not matches:
        return [("md", markdown)]

    parts: list[tuple[str, str]] = []
    cursor = 0
    for match in matches:
        before = markdown[cursor : match.start()]
        if before:
            parts.append(("md", before))
        alt = (match.group("alt") or "").strip()
        src = (match.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        cursor = match.end()

    tail = markdown[cursor:]
    if tail:
        parts.append(("md", tail))
    return parts


def render_markdown_with_local_images(markdown: str, base_dir: Path) -> None:
    markdown = normalize_markdown(markdown)
    parts = split_markdown_for_rendering(markdown)
    if len(parts) == 1 and parts[0][0] == "md":
        st.markdown(markdown, unsafe_allow_html=False)
        return

    index = 0
    while index < len(parts):
        kind, payload = parts[index]
        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            index += 1
            continue

        alt, src = payload.split("|||", 1)
        caption = None
        if index + 1 < len(parts) and parts[index + 1][0] == "md":
            next_md = parts[index + 1][1].lstrip()
            if next_md.strip():
                first_line = next_md.splitlines()[0].strip()
                caption_match = _CAPTION_LINE_RE.match(first_line)
                if caption_match:
                    caption = caption_match.group("caption").strip()
                    parts[index + 1] = ("md", "\n".join(next_md.splitlines()[1:]))

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None))
        else:
            image_path = resolve_markdown_image_path(base_dir, src)
            if image_path.exists():
                st.image(str(image_path), caption=caption or (alt or None))
            else:
                st.warning(f"Image not found: `{src}` (looked for `{image_path}`)")
        index += 1


def coerce_plan_dict(result: BlogRunResult) -> dict[str, Any] | None:
    plan = result.plan
    if not plan:
        return None
    if isinstance(plan, Plan):
        return plan.model_dump()
    if isinstance(plan, dict):
        return plan
    return None


def _strip_existing_references_section(markdown: str) -> str:
    matches = list(_REFERENCES_HEADING_RE.finditer(markdown))
    if not matches:
        return markdown.rstrip()
    return markdown[: matches[-1].start()].rstrip()


def _build_reference_lines(evidence: list[EvidenceItem]) -> list[str]:
    lines: list[str] = []
    seen_urls: set[str] = set()

    for item in evidence:
        url = (item.url or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or url in seen_urls:
            continue

        seen_urls.add(url)
        domain = parsed.netloc.removeprefix("www.")
        title = (item.title or "").strip() or domain
        lines.append(f"{len(lines) + 1}. [{title}]({url}) - {domain}")
    return lines


def _normalize_markdown_prose(markdown: str) -> str:
    markdown = _HEADING_RE.sub(r"\1 \2", markdown)
    markdown = _BULLET_RE.sub(r"\1- ", markdown)
    markdown = _ORDERED_PAREN_RE.sub(r"\1. \2", markdown)
    markdown = _LIST_RE.sub(r"\1\2 \3", markdown)
    markdown = _BLOCKQUOTE_RE.sub(r"\1 \2", markdown)
    markdown = re.sub(r"(?<!\n)\n(#{1,6}\s)", r"\n\n\1", markdown)
    markdown = re.sub(r"(?<!\n)\n(>\s)", r"\n\n\1", markdown)
    markdown = "\n".join(line.rstrip() for line in markdown.splitlines())
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown
