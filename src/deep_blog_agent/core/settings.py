"""Application settings and environment loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

from deep_blog_agent.blog_writer.contracts import (
    ImagePriceConfig,
    LLMPriceConfig,
    PricingConfig,
    SearchPriceConfig,
)


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_pricing_config() -> PricingConfig:
    return PricingConfig(
        label="Application defaults",
        openai_models={
            "gpt-4.1-mini": LLMPriceConfig(
                input_per_1m_tokens_usd=0.40,
                output_per_1m_tokens_usd=1.60,
            )
        },
        tavily_search=SearchPriceConfig(per_query_usd=0.005),
        google_image_models={
            "gemini-2.5-flash-image": ImagePriceConfig(per_image_usd=0.04),
        },
    )


class AppSettings(BaseModel):
    """Runtime settings for the application."""

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4.1-mini"
    tavily_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    google_image_model: str = "gemini-2.5-flash-image"
    default_enable_research: bool = True
    default_enable_images: bool = True
    langsmith_tracing: bool = False
    langsmith_project: Optional[str] = None
    pricing: PricingConfig = Field(default_factory=_default_pricing_config)
    outputs_dir: Path = Field(default_factory=lambda: Path("outputs"))
    legacy_blogs_dir: Path = Field(default_factory=lambda: Path("."))
    env_file: Path = Field(default_factory=lambda: Path(".env"))

    @model_validator(mode="after")
    def _ensure_default_pricing_entries(self) -> "AppSettings":
        self.pricing.openai_models.setdefault(
            self.openai_model,
            LLMPriceConfig(input_per_1m_tokens_usd=0.40, output_per_1m_tokens_usd=1.60),
        )
        self.pricing.google_image_models.setdefault(
            self.google_image_model,
            ImagePriceConfig(per_image_usd=0.04),
        )
        if self.pricing.tavily_search.per_query_usd == 0.0 and self.pricing.label == "Application defaults":
            self.pricing.tavily_search = SearchPriceConfig(per_query_usd=0.005)
        return self

    @classmethod
    def load(cls, env_file: str | Path | None = None) -> "AppSettings":
        """Load settings from the environment after loading dotenv once."""
        dotenv_path = Path(env_file) if env_file is not None else Path(".env")
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=env_file is not None)
        else:
            load_dotenv(override=False)

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
            google_image_model=os.getenv("GOOGLE_IMAGE_MODEL", "gemini-2.5-flash-image"),
            default_enable_research=_read_bool("DEEP_BLOG_AGENT_DEFAULT_ENABLE_RESEARCH", True),
            default_enable_images=_read_bool("DEEP_BLOG_AGENT_DEFAULT_ENABLE_IMAGES", True),
            langsmith_tracing=_read_bool("LANGSMITH_TRACING", False),
            langsmith_project=os.getenv("LANGSMITH_PROJECT"),
            pricing=PricingConfig(
                label=os.getenv("DEEP_BLOG_AGENT_PRICING_LABEL", "Application defaults"),
                openai_models={
                    os.getenv("OPENAI_MODEL", "gpt-4.1-mini"): LLMPriceConfig(
                        input_per_1m_tokens_usd=float(
                            os.getenv("DEEP_BLOG_AGENT_OPENAI_INPUT_PRICE_PER_1M_TOKENS_USD", "0.40")
                        ),
                        output_per_1m_tokens_usd=float(
                            os.getenv("DEEP_BLOG_AGENT_OPENAI_OUTPUT_PRICE_PER_1M_TOKENS_USD", "1.60")
                        ),
                    )
                },
                tavily_search=SearchPriceConfig(
                    per_query_usd=float(os.getenv("DEEP_BLOG_AGENT_TAVILY_SEARCH_PRICE_USD", "0.005"))
                ),
                google_image_models={
                    os.getenv("GOOGLE_IMAGE_MODEL", "gemini-2.5-flash-image"): ImagePriceConfig(
                        per_image_usd=float(os.getenv("DEEP_BLOG_AGENT_GOOGLE_IMAGE_PRICE_USD", "0.04"))
                    )
                },
            ),
            outputs_dir=Path(os.getenv("DEEP_BLOG_AGENT_OUTPUT_DIR", "outputs")),
            legacy_blogs_dir=Path(os.getenv("DEEP_BLOG_AGENT_LEGACY_BLOGS_DIR", ".")),
            env_file=dotenv_path,
        )
