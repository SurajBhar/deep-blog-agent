"""Artifact helper functions."""

from __future__ import annotations

import re
from pathlib import Path


def slugify_title(title: str) -> str:
    value = title.strip().lower()
    value = re.sub(r"[^a-z0-9 _-]+", "", value)
    value = re.sub(r"\s+", "_", value).strip("_")
    return value or "blog"


def extract_title_from_markdown(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            return title or fallback
    return fallback


def resolve_markdown_image_path(base_dir: Path, src: str) -> Path:
    cleaned = src.strip().lstrip("./")
    return (base_dir / cleaned).resolve()
