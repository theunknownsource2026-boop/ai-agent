"""
Ollama provider implementation.

Talks directly to the Ollama REST API via ``httpx`` -- no SDK required.
Models run locally so cost is always zero.
"""

from __future__ import annotations

import json
import logging
from typing import Generator, Optional, List, Dict, Any

import httpx

from agent.providers import BaseProvider, ChatResponse
from agent import config

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """Provider for locally-hosted Ollama models."""

    name = "ollama"

    def __init__(
        self,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout: float = 120.0,
    ):
        self._base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self._default_model = default_model or config.OLLAMA_LLAMA
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: list,
        model: str,
        tools: list | None,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict:
        """Build the JSON payload for /api/chat."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools
        return payload

    @staticmethod
    def _parse_tool_calls(raw_message: dict) -> Optional[List[Dict[str, Any]]]:
        """Extract tool_calls from the Ollama response message."""
        tool_calls = raw_message.get("tool_calls")
        if not tool_calls:
            return None
        parsed: list[dict] = []
        for idx, tc in enumerate(tool_calls):
            func = tc.get("function", {})
            parsed.append({
                "id": f"ollama_call_{idx}",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": json.dumps(func.get("arguments", {})),
                },
            })
        return parsed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list,
        model: str = None,
        tools: list = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat request to Ollama and return a ChatResponse."""
        model = model or self._default_model
        payload = self._build_payload(messages, model, tools, temperature, max_tokens, stream=False)

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(f"{self._base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.error("Ollama API error: %s", exc)
            raise

        message = data.get("message", {})
        content = message.get("content", "")

        # Ollama token counts
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        # Determine finish reason
        done_reason = data.get("done_reason", "stop")
        finish_reason = "tool_calls" if message.get("tool_calls") else done_reason

        return ChatResponse(
            content=content,
            model=model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=self._parse_tool_calls(message),
            finish_reason=finish_reason,
        )

    def stream_chat(
        self,
        messages: list,
        model: str = None,
        tools: list = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> Generator[str, None, ChatResponse]:
        """Stream tokens from Ollama; return final ChatResponse."""
        model = model or self._default_model
        payload = self._build_payload(messages, model, tools, temperature, max_tokens, stream=True)

        full_content: list[str] = []
        input_tokens = 0
        output_tokens = 0
        finish_reason = "stop"
        final_message: dict = {}

        try:
            with httpx.Client(timeout=self._timeout) as client:
                with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        message = chunk.get("message", {})
                        token = message.get("content", "")

                        if token:
                            full_content.append(token)
                            yield token

                        # Final chunk contains token counts and done=true
                        if chunk.get("done", False):
                            input_tokens = chunk.get("prompt_eval_count", 0)
                            output_tokens = chunk.get("eval_count", 0)
                            finish_reason = chunk.get("done_reason", "stop")
                            final_message = message

        except Exception as exc:
            logger.error("Ollama streaming error: %s", exc)
            raise

        return ChatResponse(
            content="".join(full_content),
            model=model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=self._parse_tool_calls(final_message),
            finish_reason=finish_reason,
        )

    def is_available(self) -> bool:
        """Check if Ollama is running and reachable."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return names of models currently available in Ollama."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as exc:
            logger.warning("Could not list Ollama models: %s", exc)
            return []
