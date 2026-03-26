from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_blog_agent.blog_writer.contracts import ProviderStatus, RunCostSummary, SavedBlog
from deep_blog_agent.ui.view_models import (
    aggregate_provider_costs,
    finops_rows,
    format_duration,
    format_usd,
    history_rows,
    provider_summary,
)


class UiViewModelTestCase(unittest.TestCase):
    def test_provider_summary_and_currency_format(self) -> None:
        rows = provider_summary(
            [
                ProviderStatus(
                    provider="openai",
                    state="using_session_override",
                    ready=True,
                    credential_source="session",
                    message="Ready",
                    model="gpt-4.1-mini",
                )
            ]
        )

        self.assertEqual(rows[0]["provider"], "openai")
        self.assertTrue(rows[0]["ready"])
        self.assertEqual(format_usd(1.2345), "$1.2345")
        self.assertEqual(format_usd(None), "Cost unavailable")
        self.assertEqual(format_duration(59), "59s")
        self.assertEqual(format_duration(125), "2m 5s")

    def test_history_and_finops_rows_cover_missing_costs(self) -> None:
        with_cost = SavedBlog(
            run_id="run-1",
            source="run",
            title="Run One",
            markdown_path=Path("/tmp/run-1/blog.md"),
            base_dir=Path("/tmp/run-1"),
            status="complete",
            provider_mix=["openai", "tavily"],
            cost_summary=RunCostSummary(
                available=True,
                total_estimated_cost_usd=0.1234,
                by_provider={"openai": 0.1, "tavily": 0.0234},
                search_calls=1,
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
            ),
        )
        without_cost = SavedBlog(
            run_id="legacy-1",
            source="legacy",
            title="Legacy",
            markdown_path=Path("/tmp/legacy.md"),
            base_dir=Path("/tmp"),
            status="legacy",
        )

        history = history_rows([with_cost, without_cost])
        finops = finops_rows([with_cost, without_cost])
        provider_costs = aggregate_provider_costs([with_cost, without_cost])

        self.assertEqual(history[0]["estimated_cost_label"], "$0.1234")
        self.assertEqual(history[1]["estimated_cost_label"], "Cost unavailable")
        self.assertEqual(finops[0]["estimated_cost_usd"], 0.1234)
        self.assertIsNone(finops[1]["estimated_cost_usd"])
        self.assertEqual(provider_costs["openai"], 0.1)
        self.assertEqual(provider_costs["tavily"], 0.0234)
