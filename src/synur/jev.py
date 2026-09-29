"""The only live-model seam: TypeSafe or Pi Scorer System One with explicit opt-in."""

from __future__ import annotations

import getpass
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    ScoreAnswer,
    TypeSafeClient,
    TypeSafeError,
)

PI_SCORER_BASE_URL = "https://pi-scoring.centralus.inference.ml.azure.com"
PROVIDERS = {"typesafe": ("TypeSafe", "jev-1.13.0"), "pi-scorer": ("Pi Scorer", "pi-scorer")}


class ModelCallError(RuntimeError):
    """An explicit model/configuration failure, never an empty extraction."""


@dataclass(frozen=True)
class ModelReply:
    answers: dict[str, ChoiceAnswer | NoulAnswer | ScoreAnswer]
    model: str
    usage: dict = field(default_factory=dict)
    request_id: str | None = None
    provider: str | None = None


class ModelAdapter(Protocol):
    def ask(self, state: dict, questions: dict[str, Choice | Noul]) -> ModelReply: ...


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    base_url: str | None


def _provider(provider: str | None) -> str:
    selected = provider if provider is not None else os.environ.get("SYNUR_MODEL_PROVIDER", "typesafe")
    if selected not in PROVIDERS:
        raise ModelCallError("SYNUR_MODEL_PROVIDER must be 'typesafe' or 'pi-scorer'.")
    return selected


def resolve_model_config(
    *, provider: str | None = None, model: str | None = None, base_url: str | None = None
) -> ModelConfig:
    """Resolve a profile without credentials, client construction, or network calls."""
    selected = _provider(provider)
    if model is None:
        model = os.environ.get(
            "TYPESAFE_MODEL", os.environ.get("TYPESAFE_DEFAULT_MODEL", PROVIDERS[selected][1])
        )
    if not model.strip():
        raise ModelCallError("Configure a nonempty System One model ID.")
    url = base_url if base_url is not None else os.environ.get("TYPESAFE_BASE_URL")
    if url is None and selected == "pi-scorer":
        url = PI_SCORER_BASE_URL
    if url is not None:
        invalid_url = (
            "TYPESAFE_BASE_URL must be an HTTPS URL without credentials, query, or fragment."
        )
        try:
            parsed = urlparse(url)
            valid = (
                parsed.scheme == "https" and parsed.hostname
                and not parsed.username and not parsed.password
                and not parsed.query and not parsed.fragment
                and not any(character.isspace() for character in url)
            )
            parsed.port
        except ValueError:
            raise ModelCallError(invalid_url) from None
        if not valid:
            raise ModelCallError(invalid_url)
        if selected == "pi-scorer":
            if parsed.path.rstrip("/") not in {"", "/invocations", "/v1/systemone"}:
                raise ModelCallError(
                    "Pi Scorer requires a host-only base URL. Only /invocations or "
                    "/v1/systemone endpoint URLs can be normalized to that host."
                )
            # Pi's documented SDK route is /v1/systemone, not Azure ML's /invocations.
            url = f"{parsed.scheme}://{parsed.netloc}"
    return ModelConfig(provider=selected, model=model, base_url=url)


def _credential(value: str | None, *, provider: str = "typesafe") -> str:
    if not value or not value.strip():
        name = "JEV" if provider == "typesafe" else "Pi Scorer"
        raise ModelCallError(f"Set TYPESAFE_API_KEY before enabling live {name} calls.")
    key = value.strip()
    if (
        key.lower() in {"placeholder", "cache-only", "todo", "test", "your_api_key"}
        or "replace" in key.lower()
        or "your_key" in key.lower()
        or key.startswith("<")
        or key.endswith(">")
    ):
        raise ModelCallError("TYPESAFE_API_KEY is a placeholder; live calls were not made.")
    return key


def configure_api_key(*, enabled: bool = False, provider: str | None = None) -> None:
    """Reuse an environment key or read masked input into the kernel environment only."""
    if not enabled:
        return
    selected = _provider(provider)
    try:
        _credential(os.environ.get("TYPESAFE_API_KEY"), provider=selected)
    except ModelCallError:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            try:
                key = _credential(
                    getpass.getpass(f"{PROVIDERS[selected][0]} API key (hidden): "),
                    provider=selected,
                )
            except (getpass.GetPassWarning, EOFError) as exc:
                raise ModelCallError(
                    "Masked key input is unavailable. Set TYPESAFE_API_KEY in the kernel's "
                    "environment before launching Jupyter; no key was saved."
                ) from exc
        os.environ["TYPESAFE_API_KEY"] = key


class JevAdapter:
    """Construct only inside an explicitly enabled live-run block."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not enabled:
            raise ModelCallError("Live JEV calls are disabled. Opt in explicitly to run inference.")
        config = resolve_model_config(provider=provider, model=model, base_url=base_url)
        self.provider = config.provider
        self.model = config.model
        key = _credential(
            api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY"),
            provider=self.provider,
        )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ModelCallError("Model timeout must be finite and positive.")
        self._client = TypeSafeClient(
            api_key=key, model=self.model, base_url=config.base_url, timeout=timeout
        )

    def ask(self, state: dict, questions: dict[str, Choice | Noul]) -> ModelReply:
        try:
            response = self._client.system_one(state=state, questions=questions, model=self.model)
        except TypeSafeError as exc:
            # SDK exceptions may include request content; keep persisted diagnostics body-free.
            raise ModelCallError(
                f"{PROVIDERS[self.provider][0]} {type(exc).__name__}: inference failed. "
                "Check authentication, "
                "rate limits, request size, or connectivity; this is not a negative finding."
            ) from None
        return ModelReply(
            answers=dict(response.answers),
            model=response.model,
            usage=response.usage.model_dump(mode="json"),
            request_id=response.raw_http_response.headers.get("x-typesafe-request-id"),
            provider=self.provider,
        )

    def __enter__(self) -> JevAdapter:
        return self

    def __exit__(self, *args: object) -> None:
        self._client.close()
