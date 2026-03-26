from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_blog_agent.blog_writer.contracts import (
    BlogWorkflowState,
    EvidenceItem,
    EvidencePack,
    GlobalImagePlan,
    ImageSpec,
    Plan,
    RouterDecision,
    SearchResult,
    Task,
)
from deep_blog_agent.blog_writer.nodes import BlogWriterNodes
from deep_blog_agent.core.errors import ImageGenerationError, ProviderConfigurationError
from tests.fakes import FakeImageProvider, FakeLLMProvider, FakeSearchProvider, make_provider_bundle


def base_state() -> BlogWorkflowState:
    return {
        "topic": "Agentic apps",
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": date.today().isoformat(),
        "recency_days": 7,
        "enable_research": True,
        "enable_images": True,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "generated_images": [],
        "final": "",
        "warnings": [],
        "usage_records": [],
    }


class BlogWriterNodesTestCase(unittest.TestCase):
    def test_router_node_sets_open_book_recency(self) -> None:
        llm = FakeLLMProvider(
            structured_responses=[
                RouterDecision(
                    needs_research=True,
                    mode="open_book",
                    reason="Recent topic",
                    queries=["agentic apps 2026"],
                )
            ]
        )
        nodes = BlogWriterNodes(make_provider_bundle(llm=llm))

        result = nodes.router_node(base_state())

        self.assertTrue(result["needs_research"])
        self.assertEqual(result["mode"], "open_book")
        self.assertEqual(result["recency_days"], 7)
        self.assertEqual(len(result["usage_records"]), 1)

    def test_research_node_dedupes_and_filters_old_evidence(self) -> None:
        recent_day = date.today().isoformat()
        old_day = (date.today() - timedelta(days=90)).isoformat()
        llm = FakeLLMProvider(
            structured_responses=[
                EvidencePack(
                    evidence=[
                        EvidenceItem(title="One", url="https://a.example", published_at=recent_day),
                        EvidenceItem(title="Duplicate", url="https://a.example", published_at=recent_day),
                        EvidenceItem(title="Old", url="https://old.example", published_at=old_day),
                    ]
                )
            ]
        )
        search = FakeSearchProvider(
            responses={
                "agentic apps 2026": [
                    SearchResult(title="One", url="https://a.example", snippet="A"),
                    SearchResult(title="Old", url="https://old.example", snippet="Old"),
                ]
            }
        )
        nodes = BlogWriterNodes(make_provider_bundle(llm=llm, search=search))
        state = base_state()
        state["queries"] = ["agentic apps 2026"]
        state["mode"] = "open_book"
        state["recency_days"] = 7

        result = nodes.research_node(state)

        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["evidence"][0].url, "https://a.example")
        self.assertEqual(result["warnings"], [])
        self.assertEqual(len(result["usage_records"]), 2)

    def test_research_node_returns_warning_for_missing_search_configuration(self) -> None:
        llm = FakeLLMProvider(structured_responses=[])
        search = FakeSearchProvider(
            responses={"query": ProviderConfigurationError("TAVILY_API_KEY is not set.")}
        )
        nodes = BlogWriterNodes(make_provider_bundle(llm=llm, search=search))
        state = base_state()
        state["queries"] = ["query"]

        result = nodes.research_node(state)

        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["warnings"], ["TAVILY_API_KEY is not set."])

    def test_worker_node_returns_section_markdown(self) -> None:
        llm = FakeLLMProvider(invoke_responses=["## Section Title\n\nBody"])
        nodes = BlogWriterNodes(make_provider_bundle(llm=llm))

        payload = {
            "task": Task(
                id=1,
                title="Section Title",
                goal="Teach something.",
                bullets=["One", "Two", "Three"],
                target_words=150,
            ).model_dump(),
            "plan": Plan(
                blog_title="Demo",
                audience="Developers",
                tone="Practical",
                tasks=[],
            ).model_dump(),
            "topic": "Agentic apps",
            "mode": "closed_book",
            "as_of": date.today().isoformat(),
            "recency_days": 3650,
            "evidence": [],
        }

        result = nodes.worker_node(payload)

        self.assertEqual(result["sections"], [(1, "## Section Title\n\nBody")])
        self.assertEqual(len(result["usage_records"]), 1)

    def test_generate_and_place_images_inserts_failure_block_and_warning(self) -> None:
        image = FakeImageProvider(default=ImageGenerationError("quota exceeded"))
        nodes = BlogWriterNodes(make_provider_bundle(image=image))
        state = base_state()
        state["md_with_placeholders"] = "# Demo\n\n[[IMAGE_1]]"
        state["image_specs"] = [
            ImageSpec(
                placeholder="[[IMAGE_1]]",
                filename="diagram.png",
                alt="diagram",
                caption="Diagram caption",
                prompt="Make a diagram",
            ).model_dump()
        ]

        result = nodes.generate_and_place_images(state)

        self.assertIn("[IMAGE GENERATION FAILED]", result["final"])
        self.assertEqual(result["generated_images"], [])
        self.assertEqual(len(result["warnings"]), 1)
        self.assertEqual(result["usage_records"], [])

    def test_decide_images_respects_disable_toggle(self) -> None:
        llm = FakeLLMProvider(
            structured_responses=[
                GlobalImagePlan(md_with_placeholders="# Demo", images=[]),
            ]
        )
        nodes = BlogWriterNodes(make_provider_bundle(llm=llm))
        state = base_state()
        state["enable_images"] = False
        state["merged_md"] = "# Demo"
        state["plan"] = Plan(blog_title="Demo", audience="Developers", tone="Direct", tasks=[])

        result = nodes.decide_images(state)

        self.assertEqual(result["md_with_placeholders"], "# Demo")
        self.assertEqual(result["image_specs"], [])
