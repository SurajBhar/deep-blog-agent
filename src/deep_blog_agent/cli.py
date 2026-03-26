"""Command-line entrypoint for one-shot blog generation."""

from __future__ import annotations

import argparse
from datetime import date

from deep_blog_agent.blog_writer.contracts import BlogRequest
from deep_blog_agent.blog_writer.service import build_default_blog_generation_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a blog post with Deep Blog Agent.")
    parser.add_argument("topic", help="Blog topic or title to generate.")
    parser.add_argument("--as-of", default=date.today().isoformat(), help="Reference date in YYYY-MM-DD format.")
    parser.add_argument("--no-research", action="store_true", help="Disable web research mode.")
    parser.add_argument("--no-images", action="store_true", help="Disable generated images.")
    parser.add_argument("--print-markdown", action="store_true", help="Print final markdown to stdout.")
    args = parser.parse_args()

    service = build_default_blog_generation_service()
    request = BlogRequest(
        topic=args.topic,
        as_of=date.fromisoformat(args.as_of),
        enable_research=not args.no_research,
        enable_images=not args.no_images,
    )
    result = service.run(request)

    if args.print_markdown:
        print(result.final_markdown)
    elif result.artifacts:
        print(result.artifacts.markdown_path)
