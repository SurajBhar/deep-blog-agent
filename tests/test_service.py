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
from deep_blog_agent.blog_writer.contracts import (
    BlogRequest,
    GlobalImagePlan,
    ImageSpec,
    Plan,
    RouterDecision,
    Task,
)
from deep_blog_agent.blog_writer.service import BlogGenerationService
from deep_blog_agent.core.errors import ImageGenerationError, ProviderConfigurationError
from deep_blog_agent.core.settings import AppSettings
from tests.fakes import FakeImageProvider, FakeLLMProvider, FakeSearchProvider, make_provider_bundle


class BlogGenerationServiceTestCase(unittest.TestCase):
    def test_run_persists_markdown_run_json_and_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = AppSettings(outputs_dir=Path(tmpdir) / "outputs", legacy_blogs_dir=Path(tmpdir))
            llm = FakeLLMProvider(
                invoke_responses=["## Overview\n\nBody"],
                structured_responses=[
                    RouterDecision(needs_research=False, mode="closed_book", reason="Evergreen", queries=[]),
                    Plan(
                        blog_title="Modular Agentic Apps",
                        audience="Engineers",
                        tone="Direct",
                        tasks=[
                            Task(
                                id=1,
                                title="Overview",
                                goal="Explain the architecture.",
                                bullets=["One", "Two", "Three"],
                                target_words=180,
                            )
                        ],
                    ),
                    GlobalImagePlan(
                        md_with_placeholders="# Modular Agentic Apps\n\n## Overview\n\nBody\n\n[[IMAGE_1]]",
                        images=[
                            ImageSpec(
                                placeholder="[[IMAGE_1]]",
                                filename="overview.png",
                                alt="Architecture overview",
                                caption="Workflow architecture",
                                prompt="Create a workflow diagram",
                            )
                        ],
                    ),
                ],
            )
            image = FakeImageProvider(default=b"png-data")
            service = BlogGenerationService(
                settings=settings,
                providers=make_provider_bundle(llm=llm, image=image),
                artifact_store=FileSystemArtifactStore(settings),
            )

            result = service.run(BlogRequest(topic="Modular agentic apps", as_of=date(2026, 3, 25)))

            self.assertTrue(result.artifacts)
            self.assertTrue(result.artifacts.markdown_path.exists())
            self.assertTrue(result.artifacts.run_json_path and result.artifacts.run_json_path.exists())
            self.assertTrue(result.artifacts.images_dir and (result.artifacts.images_dir / "overview.png").exists())
            self.assertIn("images/overview.png", result.final_markdown)
            self.assertTrue(result.usage_records)
            self.assertIsNotNone(result.cost_summary)
            self.assertGreater(result.cost_summary.total_estimated_cost_usd, 0.0)
            self.assertIn("openai", result.cost_summary.by_provider)
            self.assertIn("google", result.cost_summary.by_provider)

    def test_stream_surfaces_search_warning_and_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = AppSettings(outputs_dir=Path(tmpdir) / "outputs", legacy_blogs_dir=Path(tmpdir))
            llm = FakeLLMProvider(
                invoke_responses=["## Section\n\nBody"],
                structured_responses=[
                    RouterDecision(needs_research=True, mode="hybrid", reason="Needs recency", queries=["query"]),
                    Plan(
                        blog_title="Research Warning",
                        audience="Engineers",
                        tone="Direct",
                        tasks=[
                            Task(
                                id=1,
                                title="Section",
                                goal="Explain the topic.",
                                bullets=["One", "Two", "Three"],
                                target_words=150,
                            )
                        ],
                    ),
                    GlobalImagePlan(md_with_placeholders="# Research Warning\n\n## Section\n\nBody", images=[]),
                ],
            )
            search = FakeSearchProvider(
                responses={"query": ProviderConfigurationError("TAVILY_API_KEY is not set.")}
            )
            service = BlogGenerationService(
                settings=settings,
                providers=make_provider_bundle(llm=llm, search=search),
                artifact_store=FileSystemArtifactStore(settings),
            )

            events = list(service.stream(BlogRequest(topic="Research warning", as_of=date(2026, 3, 25))))

            warnings = [event for event in events if event.kind == "warning"]
            usage_events = [event for event in events if event.kind == "usage"]
            self.assertEqual(len(warnings), 1)
            self.assertIn("TAVILY_API_KEY is not set.", warnings[0].message)
            self.assertGreaterEqual(len(usage_events), 2)
            self.assertEqual(events[-1].kind, "result")

    def test_stream_surfaces_image_failure_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = AppSettings(outputs_dir=Path(tmpdir) / "outputs", legacy_blogs_dir=Path(tmpdir))
            llm = FakeLLMProvider(
                invoke_responses=["## Section\n\nBody"],
                structured_responses=[
                    RouterDecision(needs_research=False, mode="closed_book", reason="Evergreen", queries=[]),
                    Plan(
                        blog_title="Image Warning",
                        audience="Engineers",
                        tone="Direct",
                        tasks=[
                            Task(
                                id=1,
                                title="Section",
                                goal="Explain the topic.",
                                bullets=["One", "Two", "Three"],
                                target_words=150,
                            )
                        ],
                    ),
                    GlobalImagePlan(
                        md_with_placeholders="# Image Warning\n\n## Section\n\nBody\n\n[[IMAGE_1]]",
                        images=[
                            ImageSpec(
                                placeholder="[[IMAGE_1]]",
                                filename="diagram.png",
                                alt="Diagram",
                                caption="Image caption",
                                prompt="Create image",
                            )
                        ],
                    ),
                ],
            )
            image = FakeImageProvider(default=ImageGenerationError("quota exceeded"))
            service = BlogGenerationService(
                settings=settings,
                providers=make_provider_bundle(llm=llm, image=image),
                artifact_store=FileSystemArtifactStore(settings),
            )

            events = list(service.stream(BlogRequest(topic="Image warning", as_of=date(2026, 3, 25))))

            warnings = [event for event in events if event.kind == "warning"]
            self.assertEqual(len(warnings), 1)
            self.assertIn("Image generation failed", warnings[0].message)
            final_result = events[-1].payload["result"]
            self.assertIn("[IMAGE GENERATION FAILED]", final_result["final_markdown"])
            self.assertTrue(final_result["cost_summary"]["total_estimated_cost_usd"] >= 0.0)
