"""Filesystem-backed artifact store."""

from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from deep_blog_agent.artifacts.utils import extract_title_from_markdown, slugify_title
from deep_blog_agent.blog_writer.contracts import BlogArtifacts, BlogRunResult, RunCostSummary, SavedBlog
from deep_blog_agent.core.errors import ArtifactStoreError
from deep_blog_agent.core.settings import AppSettings


class FileSystemArtifactStore:
    """Save runs under outputs/ and expose legacy markdown compatibility."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def save_run(self, result: BlogRunResult, generated_images: list[dict]) -> BlogArtifacts:
        title = result.blog_title or extract_title_from_markdown(result.final_markdown, "blog")
        slug = slugify_title(title)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        run_dir = self.settings.outputs_dir / f"{stamp}_{slug}"
        markdown_path = run_dir / "blog.md"
        run_json_path = run_dir / "run.json"
        images_dir = run_dir / "images"

        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            markdown_path.write_text(result.final_markdown, encoding="utf-8")

            image_files: list[str] = []
            if generated_images:
                images_dir.mkdir(exist_ok=True)
                for image in generated_images:
                    file_path = images_dir / image["filename"]
                    file_path.write_bytes(image["bytes"])
                    image_files.append(image["filename"])

            artifacts = BlogArtifacts(
                base_dir=run_dir,
                markdown_path=markdown_path,
                run_json_path=run_json_path,
                run_dir=run_dir,
                images_dir=images_dir if images_dir.exists() else None,
                image_files=image_files,
            )
            persisted = result.model_copy(update={"artifacts": artifacts})
            run_json_path.write_text(persisted.model_dump_json(indent=2), encoding="utf-8")
            return artifacts
        except Exception as exc:
            raise ArtifactStoreError(f"Unable to save run artifacts: {exc}") from exc

    def list_runs(self, limit: int = 50, search_text: str | None = None) -> list[SavedBlog]:
        entries: list[SavedBlog] = []
        normalized_search = (search_text or "").strip().lower()

        outputs_dir = self.settings.outputs_dir
        if outputs_dir.exists():
            for run_dir in outputs_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                markdown_path = run_dir / "blog.md"
                run_json_path = run_dir / "run.json"
                if not markdown_path.exists():
                    continue
                title = run_dir.name
                request_topic = None
                provider_mix: list[str] = []
                cost_summary = None
                if run_json_path.exists():
                    try:
                        payload = json.loads(run_json_path.read_text(encoding="utf-8"))
                        title = (
                            payload.get("blog_title")
                            or payload.get("plan", {}).get("blog_title")
                            or title
                        )
                        request_topic = payload.get("request", {}).get("topic")
                        provider_mix = sorted(
                            {
                                item.get("provider")
                                for item in payload.get("usage_records", [])
                                if isinstance(item, dict) and item.get("provider")
                            }
                        )
                        if payload.get("cost_summary"):
                            cost_summary = RunCostSummary.model_validate(payload["cost_summary"])
                    except Exception:
                        title = title
                else:
                    title = extract_title_from_markdown(markdown_path.read_text(encoding="utf-8"), run_dir.name)

                if normalized_search and normalized_search not in f"{title} {request_topic or ''}".lower():
                    continue
                entries.append(
                    SavedBlog(
                        run_id=run_dir.name,
                        source="run",
                        title=title,
                        markdown_path=markdown_path,
                        run_json_path=run_json_path if run_json_path.exists() else None,
                        run_dir=run_dir,
                        base_dir=run_dir,
                        created_at=datetime.fromtimestamp(markdown_path.stat().st_mtime, UTC).isoformat(),
                        status="complete",
                        request_topic=request_topic,
                        provider_mix=provider_mix,
                        cost_summary=cost_summary,
                    )
                )

        legacy_dir = self.settings.legacy_blogs_dir
        if legacy_dir.exists():
            for markdown_path in legacy_dir.glob("*.md"):
                if markdown_path.name in {"README.md"}:
                    continue
                if normalized_search and normalized_search not in markdown_path.stem.lower():
                    continue
                entries.append(
                    SavedBlog(
                        run_id=f"legacy::{markdown_path.stem}",
                        source="legacy",
                        title=extract_title_from_markdown(
                            markdown_path.read_text(encoding="utf-8", errors="replace"),
                            markdown_path.stem,
                        ),
                        markdown_path=markdown_path,
                        base_dir=legacy_dir,
                        created_at=datetime.fromtimestamp(markdown_path.stat().st_mtime, UTC).isoformat(),
                        status="legacy",
                    )
                )

        entries.sort(key=lambda item: item.markdown_path.stat().st_mtime, reverse=True)
        return entries[:limit]

    def list_cost_history(self, limit: int = 200) -> list[SavedBlog]:
        return [entry for entry in self.list_runs(limit=limit) if entry.source == "run"]

    def read_run(self, saved_blog: SavedBlog) -> BlogRunResult:
        if saved_blog.run_json_path and saved_blog.run_json_path.exists():
            try:
                data = json.loads(saved_blog.run_json_path.read_text(encoding="utf-8"))
                return BlogRunResult.model_validate(data)
            except Exception as exc:
                raise ArtifactStoreError(f"Unable to read saved run: {exc}") from exc

        markdown = saved_blog.markdown_path.read_text(encoding="utf-8", errors="replace")
        images_dir = saved_blog.base_dir / "images"
        artifacts = BlogArtifacts(
            base_dir=saved_blog.base_dir,
            markdown_path=saved_blog.markdown_path,
            run_dir=saved_blog.run_dir,
            images_dir=images_dir if images_dir.exists() else None,
            image_files=sorted(path.name for path in images_dir.iterdir() if path.is_file()) if images_dir.exists() else [],
        )
        return BlogRunResult(
            blog_title=saved_blog.title,
            final_markdown=markdown,
            artifacts=artifacts,
        )

    def build_bundle(self, result: BlogRunResult) -> bytes:
        artifacts = self._require_artifacts(result)
        markdown_bytes = result.final_markdown.encode("utf-8")
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("blog.md", markdown_bytes)
            if artifacts.images_dir and artifacts.images_dir.exists():
                for file_path in artifacts.images_dir.rglob("*"):
                    if file_path.is_file():
                        archive.write(file_path, arcname=str(Path("images") / file_path.name))
        return buffer.getvalue()

    def build_images_bundle(self, result: BlogRunResult) -> bytes | None:
        artifacts = result.artifacts
        if not artifacts or not artifacts.images_dir or not artifacts.images_dir.exists():
            return None

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in artifacts.images_dir.rglob("*"):
                if file_path.is_file():
                    archive.write(file_path, arcname=file_path.name)
        return buffer.getvalue()

    @staticmethod
    def _require_artifacts(result: BlogRunResult) -> BlogArtifacts:
        if not result.artifacts:
            raise ArtifactStoreError("Run result does not contain artifact metadata.")
        return result.artifacts
