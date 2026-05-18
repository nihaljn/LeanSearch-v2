"""OpenAI-compatible async LLM client.

Used by reasoning mode (decompose / filter / judge) and the prove task
(prover + query generator). Each named profile under `llm.<name>` in
config.yaml supplies its own base_url, api key, model, and optional
sampling defaults; consumers pick a profile by name. Mapping roles to
profiles lives in `reasoning.*_llm` and `prove.*_llm`.
"""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI

from .config import get


def _profile_config(profile: str) -> dict[str, Any]:
    cfg = get(f"LLM_{profile.upper()}", "llm", profile, default=None)
    if not isinstance(cfg, dict):
        raise RuntimeError(
            f"llm profile {profile!r} not defined in config (expected `llm.{profile}` mapping)"
        )
    return cfg


class LLMClient:
    def __init__(
        self,
        profile: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        cfg = _profile_config(profile)
        self.profile = profile
        base_url = base_url or cfg.get("base_url")
        if api_key is None:
            if cfg.get("api_key") not in (None, ""):
                api_key = str(cfg["api_key"])
            else:
                key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
                api_key = os.environ.get(str(key_env))
                if api_key is None:
                    raise RuntimeError(
                        f"LLM api key not set for profile {profile!r}: env var {key_env} is missing"
                    )
        self.model = model or cfg.get("model")
        self.timeout = float(timeout if timeout is not None else cfg.get("timeout_s", 120))
        max_retries = int(cfg.get("max_retries", 3))
        self._temperature_default = cfg.get("temperature")
        self._top_p = cfg.get("top_p")
        self._max_tokens_default = cfg.get("max_tokens")
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=self.timeout,
            max_retries=max_retries,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if temperature is None:
            temperature = self._temperature_default if self._temperature_default is not None else 0.0
        if max_tokens is None:
            max_tokens = self._max_tokens_default
        kwargs: dict[str, Any] = {}
        if self._top_p is not None:
            kwargs["top_p"] = self._top_p
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        resp = await self._client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        return resp.choices[0].message.content or ""
