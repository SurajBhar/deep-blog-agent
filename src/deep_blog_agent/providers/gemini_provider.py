"""Gemini-backed image provider."""

from __future__ import annotations

from deep_blog_agent.blog_writer.contracts import ImageUsageRecord
from deep_blog_agent.core.errors import ImageGenerationError, ProviderConfigurationError
from deep_blog_agent.providers.interfaces import ImageGenerationResult


class GeminiImageProvider:
    """Image provider implemented with google-genai."""

    def __init__(self, *, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def generate_image(self, prompt: str, *, size: str = "1024x1024", quality: str = "medium") -> ImageGenerationResult:
        if not self.api_key:
            raise ProviderConfigurationError("GOOGLE_API_KEY is not set.")

        try:
            from google import genai
            from google.genai import types
        except Exception as exc:  # pragma: no cover - import guard
            raise ImageGenerationError(f"Unable to import google-genai: {exc}") from exc

        try:
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_ONLY_HIGH",
                        )
                    ],
                ),
            )
        except Exception as exc:
            raise ImageGenerationError(str(exc)) from exc

        parts = getattr(response, "parts", None)
        if not parts and getattr(response, "candidates", None):
            try:
                parts = response.candidates[0].content.parts
            except Exception:
                parts = None

        if not parts:
            raise ImageGenerationError("No image content returned by Gemini.")

        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                image_bytes = inline_data.data
                return ImageGenerationResult(
                    image_bytes=image_bytes,
                    usage=ImageUsageRecord(
                        provider="google",
                        step="image_generation",
                        model=self.model,
                        image_count=1,
                        size=size,
                        quality=quality,
                        output_bytes=len(image_bytes),
                    ),
                )

        raise ImageGenerationError("No inline image bytes found in Gemini response.")
