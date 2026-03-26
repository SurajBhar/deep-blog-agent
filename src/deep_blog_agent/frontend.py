"""Package-native Streamlit entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    SRC_ROOT = Path(__file__).resolve().parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from deep_blog_agent.ui.app import main as app_main
else:
    from .ui.app import main as app_main


def main() -> None:
    """Run the Streamlit app body."""
    app_main()


def run() -> None:
    """Launch Streamlit against this package entry module."""
    from streamlit.web.cli import main as streamlit_main

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    raise SystemExit(streamlit_main())


if __name__ == "__main__":
    main()
