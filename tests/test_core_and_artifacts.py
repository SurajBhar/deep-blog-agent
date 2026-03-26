from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_blog_agent.artifacts.store import FileSystemArtifactStore
from deep_blog_agent.artifacts.utils import extract_title_from_markdown, resolve_markdown_image_path, slugify_title
from deep_blog_agent.blog_writer.contracts import BlogRequest, BlogRunResult, EvidenceItem, SessionRuntimeConfig
from deep_blog_agent.core.runtime import resolve_runtime
from deep_blog_agent.core.settings import AppSettings
from deep_blog_agent.ui.renderers import build_renderable_blog_markdown, normalize_markdown, split_markdown_for_rendering


class SettingsAndArtifactsTestCase(unittest.TestCase):
    def test_settings_loads_values_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=openai-key",
                        "OPENAI_MODEL=test-model",
                        "TAVILY_API_KEY=tavily-key",
                        "GOOGLE_API_KEY=google-key",
                        "LANGSMITH_API_KEY=langsmith-key",
                        "LANGSMITH_PROJECT=langsmith-project",
                        "GOOGLE_IMAGE_MODEL=image-model",
                        "DEEP_BLOG_AGENT_OUTPUT_DIR=custom_outputs",
                        "DEEP_BLOG_AGENT_LEGACY_BLOGS_DIR=legacy_blogs",
                    ]
                ),
                encoding="utf-8",
            )

            settings = AppSettings.load(env_file=env_path)

            self.assertEqual(settings.openai_api_key, "openai-key")
            self.assertEqual(settings.openai_model, "test-model")
            self.assertEqual(settings.tavily_api_key, "tavily-key")
            self.assertEqual(settings.google_api_key, "google-key")
            self.assertEqual(settings.langsmith_api_key, "langsmith-key")
            self.assertEqual(settings.langsmith_project, "langsmith-project")
            self.assertEqual(settings.google_image_model, "image-model")
            self.assertEqual(settings.outputs_dir, Path("custom_outputs"))
            self.assertEqual(settings.legacy_blogs_dir, Path("legacy_blogs"))
            self.assertTrue(settings.default_enable_research)
            self.assertTrue(settings.default_enable_images)

    def test_runtime_resolution_prefers_session_overrides(self) -> None:
        settings = AppSettings(
            openai_api_key="deployment-openai",
            openai_model="deployment-model",
            tavily_api_key="deployment-tavily",
            google_api_key="deployment-google",
            google_image_model="deployment-image",
        )
        overrides = SessionRuntimeConfig(
            openai_api_key="session-openai",
            openai_model="session-model",
            google_image_model="session-image",
            default_enable_research=False,
        )

        resolved = resolve_runtime(settings, overrides)

        self.assertEqual(resolved.settings.openai_api_key, "session-openai")
        self.assertEqual(resolved.settings.openai_model, "session-model")
        self.assertEqual(resolved.settings.google_image_model, "session-image")
        self.assertFalse(resolved.config.default_enable_research)
        self.assertEqual(resolved.config.credential_sources["openai"], "session")
        self.assertEqual(resolved.config.credential_sources["tavily"], "deployment")

    def test_slug_and_markdown_helpers(self) -> None:
        self.assertEqual(slugify_title("  Hello, World!  "), "hello_world")
        self.assertEqual(extract_title_from_markdown("# Demo\n\nBody", "fallback"), "Demo")
        self.assertEqual(resolve_markdown_image_path(Path("/tmp/demo"), "./images/a.png"), Path("/tmp/demo/images/a.png").resolve())
        self.assertEqual(
            split_markdown_for_rendering("Text\n![Alt](images/test.png)\n*Caption*"),
            [("md", "Text\n"), ("img", "Alt|||images/test.png"), ("md", "\n*Caption*")],
        )
        self.assertEqual(normalize_markdown("##Heading\n• item"), "## Heading\n- item\n")
        self.assertEqual(normalize_markdown("```python\n##Heading\n```"), "```python\n##Heading\n```\n")
        self.assertEqual(normalize_markdown("1)First\n>quote\nBad\x19s"), "1. First\n\n> quote\nBad's\n")
        self.assertEqual(normalize_markdown("```python\nprint(1)"), "```python\nprint(1)\n```\n")

    def test_renderable_blog_markdown_appends_references(self) -> None:
        markdown = "# Demo\n\nBody\n\n## References\n\nOld"
        evidence = [
            EvidenceItem(title="Example Source", url="https://example.com/post", source="example"),
            EvidenceItem(title="Broken", url="notaurl"),
            EvidenceItem(title="Example Source", url="https://example.com/post", source="example"),
        ]

        rendered = build_renderable_blog_markdown(markdown, evidence)

        self.assertIn("## References", rendered)
        self.assertIn("[Example Source](https://example.com/post) - example.com", rendered)
        self.assertNotIn("Old", rendered)

    def test_artifact_store_saves_outputs_under_outputs_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings = AppSettings(outputs_dir=root / "outputs", legacy_blogs_dir=root)
            store = FileSystemArtifactStore(settings)
            result = BlogRunResult(
                request=BlogRequest(topic="Example", as_of=date(2026, 3, 25)),
                blog_title="Example Blog",
                final_markdown="# Example Blog\n\nBody",
            )

            artifacts = store.save_run(
                result,
                [{"filename": "diagram.png", "bytes": b"abc", "alt": "Diagram", "caption": "Caption"}],
            )

            self.assertTrue(artifacts.run_dir)
            self.assertTrue(artifacts.markdown_path.exists())
            self.assertTrue(artifacts.run_json_path and artifacts.run_json_path.exists())
            self.assertTrue(artifacts.images_dir and artifacts.images_dir.exists())
            self.assertTrue((artifacts.images_dir / "diagram.png").exists())
            self.assertIn("example_blog", artifacts.run_dir.name)

    def test_artifact_store_does_not_persist_session_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings = AppSettings(outputs_dir=root / "outputs", legacy_blogs_dir=root)
            store = FileSystemArtifactStore(settings)
            result = BlogRunResult(
                request=BlogRequest(
                    topic="Secrets",
                    as_of=date(2026, 3, 25),
                    runtime_overrides=SessionRuntimeConfig(
                        openai_api_key="session-openai-secret",
                        tavily_api_key="session-tavily-secret",
                        google_api_key="session-google-secret",
                        langsmith_api_key="session-langsmith-secret",
                    ),
                ),
                blog_title="Secrets",
                final_markdown="# Secrets\n\nBody",
            )

            artifacts = store.save_run(result, [])
            run_json = artifacts.run_json_path.read_text(encoding="utf-8")

            self.assertNotIn("session-openai-secret", run_json)
            self.assertNotIn("session-tavily-secret", run_json)
            self.assertNotIn("session-google-secret", run_json)
            self.assertNotIn("session-langsmith-secret", run_json)

    def test_legacy_markdown_is_listed_and_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "legacy_blog.md").write_text("# Legacy Blog\n\nHello", encoding="utf-8")
            images_dir = root / "images"
            images_dir.mkdir()
            (images_dir / "legacy.png").write_bytes(b"png")

            settings = AppSettings(outputs_dir=root / "outputs", legacy_blogs_dir=root)
            store = FileSystemArtifactStore(settings)

            saved = store.list_runs(limit=10)
            legacy_entries = [entry for entry in saved if entry.source == "legacy"]
            self.assertEqual(len(legacy_entries), 1)
            loaded = store.read_run(legacy_entries[0])
            self.assertEqual(loaded.blog_title, "Legacy Blog")
            self.assertIn("Hello", loaded.final_markdown)
            self.assertEqual(loaded.artifacts.image_files, ["legacy.png"])

    def test_list_runs_handles_old_run_without_cost_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "outputs" / "20260325_000000_demo"
            run_dir.mkdir(parents=True)
            (run_dir / "blog.md").write_text("# Demo\n\nBody", encoding="utf-8")
            (run_dir / "run.json").write_text(
                '{"blog_title": "Demo", "final_markdown": "# Demo\\n\\nBody"}',
                encoding="utf-8",
            )
            store = FileSystemArtifactStore(AppSettings(outputs_dir=root / "outputs", legacy_blogs_dir=root))

            saved = store.list_runs(limit=10)

            self.assertEqual(saved[0].title, "Demo")
            self.assertIsNone(saved[0].cost_summary)
