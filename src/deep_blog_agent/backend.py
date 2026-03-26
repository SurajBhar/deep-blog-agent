"""Package-native backend exports."""

from __future__ import annotations

from .blog_writer.service import build_default_blog_generation_service

_service = build_default_blog_generation_service()
app = _service.graph


def get_service():
    """Return the default configured blog-generation service."""
    return _service
