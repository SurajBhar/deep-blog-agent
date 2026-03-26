from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_blog_agent import backend, frontend
from deep_blog_agent.ui import app
from deep_blog_agent.ui import pages, view_models


class UiSmokeTestCase(unittest.TestCase):
    def test_ui_module_imports(self) -> None:
        self.assertTrue(callable(app.main))
        self.assertTrue(callable(frontend.main))
        self.assertTrue(callable(frontend.run))
        self.assertIsNotNone(backend.app)
        self.assertTrue(callable(pages.render_home_page))
        self.assertTrue(callable(pages.render_prompt_examples_page))
        self.assertTrue(callable(view_models.history_rows))
