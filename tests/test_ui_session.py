from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_blog_agent.blog_writer.contracts import SavedBlog
from deep_blog_agent.ui import pages, session


class UiSessionHelpersTestCase(unittest.TestCase):
    def tearDown(self) -> None:
        session.register_pages({})

    def test_ensure_defaults_sets_home_prompt_state(self) -> None:
        fake_st = SimpleNamespace(session_state={})

        with patch.object(session, "st", fake_st):
            session.ensure_defaults()

        self.assertEqual(fake_st.session_state[session.HOME_TOPIC_INPUT_KEY], "")
        self.assertEqual(fake_st.session_state[session.BLOG_PROMPT_KEY], "")
        self.assertEqual(fake_st.session_state[session.ACTIVE_PAGE_KEY], session.PAGE_HOME)
        self.assertEqual(session.HOME_TOPIC_INPUT_KEY, session.BLOG_PROMPT_KEY)

    def test_navigate_to_switches_registered_page(self) -> None:
        fake_st = SimpleNamespace(session_state={}, switch_page=Mock())
        target_page = object()

        with patch.object(session, "st", fake_st):
            session.ensure_defaults()
            session.register_pages({session.PAGE_HOME: target_page})

            did_navigate = session.navigate_to(session.PAGE_HOME)

        self.assertTrue(did_navigate)
        self.assertEqual(fake_st.session_state[session.ACTIVE_PAGE_KEY], session.PAGE_HOME)
        fake_st.switch_page.assert_called_once_with(target_page)

    def test_open_home_with_prompt_loads_prompt_and_navigates_home(self) -> None:
        with patch.object(pages.session, "set_blog_prompt") as set_blog_prompt, patch.object(
            pages.session, "navigate_to", return_value=True
        ) as navigate_to:
            pages._open_home_with_prompt("Prompt to reuse")

        set_blog_prompt.assert_called_once_with("Prompt to reuse")
        navigate_to.assert_called_once_with(session.PAGE_HOME)

    def test_resolve_selected_blog_flags_missing_run(self) -> None:
        saved_runs = [
            SavedBlog(
                run_id="run-1",
                source="run",
                title="Run One",
                markdown_path=Path("/tmp/run-1/blog.md"),
                base_dir=Path("/tmp/run-1"),
            )
        ]

        with patch.object(pages.session, "get_selected_run_id", return_value="missing-run"):
            selected_blog, missing_selection = pages._resolve_selected_blog(saved_runs)

        self.assertTrue(missing_selection)
        self.assertIsNotNone(selected_blog)
        self.assertEqual(selected_blog.run_id, "run-1")
