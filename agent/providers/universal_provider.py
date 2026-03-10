"""
Universal OpenAI-compatible provider.

One class that talks to ANY provider using the OpenAI SDK's base_url
parameter. Groq, Mistral, DeepSeek, Cerebras, Gemini, OpenRouter,
xAI, Anthropic, Together AI, Cohere -- they all speak
the same OpenAI chat completions format.
"""

from __future__ import annotations

import json
import logging
from typing import Generator, Optional, List, Dict, Any

from openai import OpenAI

from agent.providers import BaseProvider, ChatResponse

logger = logging.getLogger(__name__)


class UniversalProvider(BaseProvider):
    """Talks to any OpenAI-compatible API endpoint."""

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        default_model: str,
        supports_tools: bool = True,
        is_local: bool = False,
    ):
        self.name = name
        self._api_key = api_key
        self._base_url = base_url
        self._default_model = default_model
        self._supports_tools = supports_tools
        self._is_local = is_local
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        """Lazy-init the OpenAI client pointed at our base_url."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    @staticmethod
    def _parse_tool_calls(raw_tool_calls) -> Optional[List[Dict[str, Any]]]:
        """Convert SDK tool-call objects to plain dicts with type field."""
        if not raw_tool_calls:
            return None
        parsed: list[dict] = []
        for tc in raw_tool_calls:
            entry = {}
            if hasattr(tc, "id"):
                entry["id"] = tc.id
                entry["type"] = "function"
                entry["function"] = {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            elif isinstance(tc, dict):
                entry = dict(tc)
                entry.setdefault("type", "function")
            else:
                continue
            parsed.append(entry)
        return parsed if parsed else None

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
        """Send a chat completion request."""
        model = model or self._default_model
        client = self._get_client()

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Only send tools if this provider supports them
        if tools and self._supports_tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")

        try:
            response = client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            logger.error("%s API error: %s", self.name, exc)
            raise

        choice = response.choices[0]
        usage = response.usage

        input_tokens = 0
        output_tokens = 0
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0

        return ChatResponse(
            content=choice.message.content or "",
            model=getattr(response, "model", model),
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=self._parse_tool_calls(
                getattr(choice.message, "tool_calls", None)
            ),
            finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
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
        """Stream tokens one at a time."""
        model = model or self._default_model
        client = self._get_client()

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if tools and self._supports_tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")

        if self.name in ("openai", "groq", "deepseek"):
            request_kwargs["stream_options"] = {"include_usage": True}

        try:
            stream = client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            logger.error("%s streaming error: %s", self.name, exc)
            raise

        full_content: list[str] = []
        finish_reason = "stop"
        input_tokens = 0
        output_tokens = 0
        tool_call_chunks: dict[int, dict] = {}

        for chunk in stream:
            if hasattr(chunk, "usage") and chunk.usage:
                input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {
                            "id": getattr(tc_delta, "id", "") or "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    entry = tool_call_chunks[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            entry["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            entry["function"]["arguments"] += tc_delta.function.arguments

            if hasattr(delta, "content") and delta.content:
                full_content.append(delta.content)
                yield delta.content

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        parsed_tools = (
            [tool_call_chunks[i] for i in sorted(tool_call_chunks)]
            if tool_call_chunks
            else None
        )

        return ChatResponse(
            content="".join(full_content),
            model=model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=parsed_tools,
            finish_reason=finish_reason,
        )

    def is_available(self) -> bool:
        """Check if this provider has a valid API key or is local."""
        if self._is_local:
            try:
                import httpx
                base = self._base_url.replace("/v1", "")
                resp = httpx.get(f"{base}/api/tags", timeout=2)
                return resp.status_code == 200
            except Exception:
                return False
        return bool(self._api_key and self._api_key.strip())

    def __repr__(self) -> str:
        return f"<UniversalProvider({self.name}) available={self.is_available()}>"
