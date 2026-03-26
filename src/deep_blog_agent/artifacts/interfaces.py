"""Artifact storage contract."""

from __future__ import annotations

from typing import Protocol

from deep_blog_agent.blog_writer.contracts import BlogArtifacts, BlogRunResult, SavedBlog


class ArtifactStore(Protocol):
    """Artifact persistence and retrieval contract."""

    def save_run(self, result: BlogRunResult, generated_images: list[dict]) -> BlogArtifacts:
        """Persist a completed run and return its artifact metadata."""

    def list_runs(self, limit: int = 50, search_text: str | None = None) -> list[SavedBlog]:
        """List saved runs plus legacy markdown files."""

    def list_cost_history(self, limit: int = 200) -> list[SavedBlog]:
        """List recent run metadata intended for cost and FinOps views."""

    def read_run(self, saved_blog: SavedBlog) -> BlogRunResult:
        """Load a run result from saved metadata."""

    def build_bundle(self, result: BlogRunResult) -> bytes:
        """Build a zip bundle containing markdown and any images."""

    def build_images_bundle(self, result: BlogRunResult) -> bytes | None:
        """Build a zip bundle containing only images, if any."""
