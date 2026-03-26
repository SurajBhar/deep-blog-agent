from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deep_blog_agent.blog_writer.contracts import PromptMessage, RouterDecision
from deep_blog_agent.core.errors import ProviderConfigurationError
from deep_blog_agent.providers.gemini_provider import GeminiImageProvider
from deep_blog_agent.providers.openai_provider import OpenAIChatProvider
from deep_blog_agent.providers.tavily_provider import TavilySearchProvider


class ProviderAdaptersTestCase(unittest.TestCase):
    def test_openai_provider_supports_plain_and_structured_calls(self) -> None:
        class FakeChatOpenAI:
            def __init__(self, model, api_key):
                self.model = model
                self.api_key = api_key
                self.schema = None
                self.include_raw = False

            def with_structured_output(self, schema, include_raw=False):
                clone = FakeChatOpenAI(self.model, self.api_key)
                clone.schema = schema
                clone.include_raw = include_raw
                return clone

            def invoke(self, messages):
                if self.schema:
                    parsed = self.schema(
                        needs_research=False,
                        mode="closed_book",
                        reason="Evergreen topic",
                        queries=[],
                        max_results_per_query=5,
                    )
                    raw = SimpleNamespace(
                        content="structured",
                        usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
                    )
                    if self.include_raw:
                        return {"parsed": parsed, "raw": raw}
                    return parsed
                return SimpleNamespace(
                    content="hello world",
                    usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

        fake_langchain_openai = types.ModuleType("langchain_openai")
        fake_langchain_openai.ChatOpenAI = FakeChatOpenAI

        fake_messages = types.ModuleType("langchain_core.messages")

        class FakeSystemMessage:
            def __init__(self, content):
                self.content = content

        class FakeHumanMessage:
            def __init__(self, content):
                self.content = content

        fake_messages.SystemMessage = FakeSystemMessage
        fake_messages.HumanMessage = FakeHumanMessage

        with patch.dict(
            sys.modules,
            {
                "langchain_openai": fake_langchain_openai,
                "langchain_core.messages": fake_messages,
            },
        ):
            provider = OpenAIChatProvider(api_key="key", model="model")
            text = provider.invoke([PromptMessage(role="user", content="Hi")])
            structured = provider.invoke_structured([PromptMessage(role="user", content="Hi")], RouterDecision)

        self.assertEqual(text.text, "hello world")
        self.assertEqual(text.usage.total_tokens, 15)
        self.assertEqual(structured.value.mode, "closed_book")
        self.assertEqual(structured.usage.total_tokens, 30)

    def test_tavily_provider_normalizes_results(self) -> None:
        fake_tavily = types.ModuleType("tavily")

        class FakeClient:
            def __init__(self, api_key):
                self.api_key = api_key

            def search(self, query, max_results):
                return {
                    "results": [
                        {
                            "title": "Result",
                            "url": "https://example.com",
                            "content": "Summary",
                            "published_date": "2026-03-20",
                            "source": "example",
                        }
                    ]
                }

        fake_tavily.TavilyClient = FakeClient

        with patch.dict(sys.modules, {"tavily": fake_tavily}):
            provider = TavilySearchProvider(api_key="key")
            result = provider.search("agentic apps", max_results=3)

        self.assertEqual(len(result.results), 1)
        self.assertEqual(result.results[0].url, "https://example.com")
        self.assertEqual(result.results[0].snippet, "Summary")
        self.assertEqual(result.usage.result_count, 1)

    def test_gemini_provider_returns_inline_bytes(self) -> None:
        fake_google = types.ModuleType("google")
        fake_genai = types.ModuleType("google.genai")

        class FakeInlineData:
            def __init__(self, data):
                self.data = data

        class FakePart:
            def __init__(self, data):
                self.inline_data = FakeInlineData(data)

        class FakeResponse:
            def __init__(self, data):
                self.parts = [FakePart(data)]

        class FakeClient:
            def __init__(self, api_key):
                self.api_key = api_key
                self.models = self

            def generate_content(self, model, contents, config):
                del model, contents, config
                return FakeResponse(b"image-bytes")

        class FakeGenerateContentConfig:
            def __init__(self, response_modalities, safety_settings):
                self.response_modalities = response_modalities
                self.safety_settings = safety_settings

        class FakeSafetySetting:
            def __init__(self, category, threshold):
                self.category = category
                self.threshold = threshold

        fake_genai.Client = FakeClient
        fake_genai.types = types.SimpleNamespace(
            GenerateContentConfig=FakeGenerateContentConfig,
            SafetySetting=FakeSafetySetting,
        )
        fake_google.genai = fake_genai

        with patch.dict(sys.modules, {"google": fake_google, "google.genai": fake_genai}):
            provider = GeminiImageProvider(api_key="key", model="image-model")
            image_result = provider.generate_image("Make a diagram")

        self.assertEqual(image_result.image_bytes, b"image-bytes")
        self.assertEqual(image_result.usage.output_bytes, len(b"image-bytes"))

    def test_missing_provider_configuration_raises(self) -> None:
        with self.assertRaises(ProviderConfigurationError):
            OpenAIChatProvider(api_key=None, model="model").invoke([PromptMessage(role="user", content="Hi")])
        with self.assertRaises(ProviderConfigurationError):
            TavilySearchProvider(api_key=None).search("query")
        with self.assertRaises(ProviderConfigurationError):
            GeminiImageProvider(api_key=None, model="model").generate_image("prompt")
