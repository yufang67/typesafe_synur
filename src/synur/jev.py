"""The only live-model seam: TypeSafe System One with explicit opt-in."""

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


class ModelCallError(RuntimeError):
    """An explicit model/configuration failure, never an empty extraction."""


@dataclass(frozen=True)
class ModelReply:
    answers: dict[str, ChoiceAnswer | NoulAnswer | ScoreAnswer]
    model: str
    usage: dict = field(default_factory=dict)
    request_id: str | None = None


class ModelAdapter(Protocol):
    def ask(self, state: dict, questions: dict[str, Choice | Noul]) -> ModelReply: ...


def _credential(value: str | None) -> str:
    if not value or not value.strip():
        raise ModelCallError("Set TYPESAFE_API_KEY before enabling live JEV calls.")
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


def configure_api_key(*, enabled: bool = False) -> None:
    """Reuse an environment key or read masked input into the kernel environment only."""
    if not enabled:
        return
    try:
        _credential(os.environ.get("TYPESAFE_API_KEY"))
    except ModelCallError:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            try:
                key = _credential(getpass.getpass("TypeSafe API key (hidden): "))
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
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not enabled:
            raise ModelCallError("Live JEV calls are disabled. Opt in explicitly to run inference.")
        key = _credential(api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY"))
        self.model = model or os.environ.get("TYPESAFE_MODEL", "jev-1.13.0")
        if not self.model.strip():
            raise ModelCallError("Configure a nonempty JEV model ID.")
        url = base_url if base_url is not None else os.environ.get("TYPESAFE_BASE_URL")
        if url is not None:
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
                raise ModelCallError("TYPESAFE_BASE_URL must be an HTTPS URL without credentials.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ModelCallError("Model timeout must be finite and positive.")
        self._client = TypeSafeClient(api_key=key, model=self.model, base_url=url, timeout=timeout)

    def ask(self, state: dict, questions: dict[str, Choice | Noul]) -> ModelReply:
        try:
            response = self._client.system_one(state=state, questions=questions, model=self.model)
        except TypeSafeError as exc:
            # SDK exceptions may include request content; keep persisted diagnostics body-free.
            raise ModelCallError(
                f"TypeSafe {type(exc).__name__}: inference failed. Check authentication, "
                "rate limits, request size, or connectivity; this is not a negative finding."
            ) from exc
        return ModelReply(
            answers=dict(response.answers),
            model=response.model,
            usage=response.usage.model_dump(mode="json"),
            request_id=response.raw_http_response.headers.get("x-typesafe-request-id"),
        )

    def __enter__(self) -> JevAdapter:
        return self

    def __exit__(self, *args: object) -> None:
        self._client.close()
