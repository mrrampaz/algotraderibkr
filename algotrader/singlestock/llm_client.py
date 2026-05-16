"""Anthropic Claude client for the investigation agents.

Each agent passes a system prompt (cacheable, large, static) plus a
small user payload (the day's data). We use Anthropic's prompt caching
to keep cost low — typical morning run is 3 LLM calls totaling well
under $0.20.

Failures are non-fatal: callers receive None and must fall back to
deterministic-only logic (which natively produces a cash-default
thesis when no LLM signal is present).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


PROMPTS_DIR = Path(__file__).parent / "prompts"


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._timeout = timeout_seconds
        self._log = logger.bind(component="singlestock_llm")
        self._client: Any | None = None
        if self._api_key:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self._api_key, timeout=timeout_seconds)
            except Exception:
                self._log.exception("singlestock_llm_init_failed")
                self._client = None
        else:
            self._log.warning("singlestock_llm_no_api_key")

    @property
    def available(self) -> bool:
        return self._client is not None

    @staticmethod
    def load_prompt(name: str) -> str:
        path = PROMPTS_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"prompt not found: {path}")
        return path.read_text(encoding="utf-8")

    def call_json(
        self,
        model: str,
        system_prompt: str,
        user_payload: dict[str, Any] | str,
        max_tokens: int = 1024,
        agent_name: str = "agent",
    ) -> dict[str, Any] | None:
        """Call Claude expecting a JSON object response.

        - system_prompt is cached (ephemeral cache_control).
        - user_payload is sent uncached as JSON.
        - The response is parsed leniently: we strip code fences and
          extract the first {...} block.
        """
        if self._client is None:
            self._log.warning("singlestock_llm_unavailable", agent=agent_name)
            return None

        if isinstance(user_payload, dict):
            user_text = json.dumps(user_payload, default=str, indent=2)
        else:
            user_text = str(user_payload)

        try:
            resp = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_text}],
                    }
                ],
            )
        except Exception:
            self._log.exception("singlestock_llm_call_failed", agent=agent_name, model=model)
            return None

        try:
            text = "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            )
        except Exception:
            self._log.exception("singlestock_llm_response_extract_failed", agent=agent_name)
            return None

        parsed = self._extract_json(text)
        if parsed is None:
            self._log.warning(
                "singlestock_llm_json_parse_failed",
                agent=agent_name,
                preview=text[:200],
            )
            return None

        usage = getattr(resp, "usage", None)
        self._log.info(
            "singlestock_llm_call_ok",
            agent=agent_name,
            model=model,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            cache_read=getattr(usage, "cache_read_input_tokens", None),
            cache_creation=getattr(usage, "cache_creation_input_tokens", None),
        )
        return parsed

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        if not text:
            return None
        # Strip markdown code fences if present.
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        else:
            # First {...} block in the text.
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            text = text[start : end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
