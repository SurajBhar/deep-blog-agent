"""Runtime configuration resolution for deployment defaults and session overrides."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from deep_blog_agent.blog_writer.contracts import (
    ImagePriceConfig,
    LLMPriceConfig,
    PricingConfig,
    ProviderStatus,
    ResolvedRuntimeConfig,
    SearchPriceConfig,
    SessionRuntimeConfig,
)
from deep_blog_agent.core.settings import AppSettings


@dataclass(frozen=True)
class ResolvedRuntime:
    settings: AppSettings
    config: ResolvedRuntimeConfig
    provider_statuses: list[ProviderStatus]


def resolve_runtime(
    settings: AppSettings,
    overrides: SessionRuntimeConfig | None = None,
    *,
    validate_providers: bool = False,
) -> ResolvedRuntime:
    """Resolve the effective runtime settings for one app session or run."""

    session_config = overrides or SessionRuntimeConfig()
    openai_api_key, openai_source = _resolve_secret(settings.openai_api_key, session_config.openai_api_key)
    tavily_api_key, tavily_source = _resolve_secret(settings.tavily_api_key, session_config.tavily_api_key)
    google_api_key, google_source = _resolve_secret(settings.google_api_key, session_config.google_api_key)
    langsmith_api_key, _langsmith_source = _resolve_secret(settings.langsmith_api_key, session_config.langsmith_api_key)

    openai_model = session_config.openai_model or settings.openai_model
    google_image_model = session_config.google_image_model or settings.google_image_model
    pricing = _normalize_pricing(
        session_config.pricing.model_copy(deep=True) if session_config.pricing else settings.pricing.model_copy(deep=True),
        openai_model=openai_model,
        google_image_model=google_image_model,
    )

    merged_settings = settings.model_copy(
        update={
            "openai_api_key": openai_api_key,
            "tavily_api_key": tavily_api_key,
            "google_api_key": google_api_key,
            "langsmith_api_key": langsmith_api_key,
            "openai_model": openai_model,
            "google_image_model": google_image_model,
            "default_enable_research": (
                session_config.default_enable_research
                if session_config.default_enable_research is not None
                else settings.default_enable_research
            ),
            "default_enable_images": (
                session_config.default_enable_images
                if session_config.default_enable_images is not None
                else settings.default_enable_images
            ),
            "langsmith_tracing": (
                session_config.langsmith_tracing
                if session_config.langsmith_tracing is not None
                else settings.langsmith_tracing
            ),
            "langsmith_project": (
                session_config.langsmith_project
                if session_config.langsmith_project is not None
                else settings.langsmith_project
            ),
            "pricing": pricing,
        }
    )

    resolved_config = ResolvedRuntimeConfig(
        openai_model=openai_model,
        google_image_model=google_image_model,
        default_enable_research=merged_settings.default_enable_research,
        default_enable_images=merged_settings.default_enable_images,
        langsmith_tracing=merged_settings.langsmith_tracing,
        langsmith_project=merged_settings.langsmith_project,
        pricing=pricing,
        credential_sources={
            "openai": openai_source,
            "tavily": tavily_source,
            "google": google_source,
        },
    )

    provider_statuses = [
        _provider_status(
            provider="openai",
            model=openai_model,
            credential_source=openai_source,
            has_key=bool(openai_api_key),
            validate=validate_providers,
            module_names=["langchain_openai", "langchain_core.messages"],
        ),
        _provider_status(
            provider="tavily",
            model=None,
            credential_source=tavily_source,
            has_key=bool(tavily_api_key),
            validate=validate_providers,
            module_names=["tavily"],
        ),
        _provider_status(
            provider="google",
            model=google_image_model,
            credential_source=google_source,
            has_key=bool(google_api_key),
            validate=validate_providers,
            module_names=["google.genai", "google"],
        ),
    ]

    return ResolvedRuntime(settings=merged_settings, config=resolved_config, provider_statuses=provider_statuses)


def _resolve_secret(deployment_value: str | None, session_value: str | None) -> tuple[str | None, str]:
    normalized_session = (session_value or "").strip() or None
    normalized_deployment = (deployment_value or "").strip() or None
    if normalized_session:
        return normalized_session, "session"
    if normalized_deployment:
        return normalized_deployment, "deployment"
    return None, "missing"


def _normalize_pricing(pricing: PricingConfig, *, openai_model: str, google_image_model: str) -> PricingConfig:
    pricing.openai_models.setdefault(openai_model, LLMPriceConfig())
    pricing.google_image_models.setdefault(google_image_model, ImagePriceConfig())
    if pricing.tavily_search is None:
        pricing.tavily_search = SearchPriceConfig()
    return pricing


def _provider_status(
    *,
    provider: str,
    model: str | None,
    credential_source: str,
    has_key: bool,
    validate: bool,
    module_names: list[str],
) -> ProviderStatus:
    if not has_key:
        return ProviderStatus(
            provider=provider,
            state="missing",
            ready=False,
            credential_source="missing",
            model=model,
            message="Missing API key.",
        )

    state = "using_session_override" if credential_source == "session" else "using_deployment_default"
    message = "Credentials available."
    ready = True
    if validate:
        try:
            for module_name in module_names:
                import_module(module_name)
            message = "Credentials available and local adapter imports succeeded."
        except Exception as exc:
            state = "validation_failed"
            ready = False
            message = f"Validation failed: {exc}"

    return ProviderStatus(
        provider=provider,
        state=state,
        ready=ready,
        credential_source=credential_source if credential_source in {"session", "deployment"} else "missing",
        model=model,
        message=message,
    )
