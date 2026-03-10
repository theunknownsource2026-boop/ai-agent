"""
OpenAI provider implementation.

Wraps the official ``openai`` Python SDK and exposes the unified
BaseProvider interface for GPT-4o / GPT-4o-mini.
"""

from __future__ import annotations

import json
import logging
from typing import Generator, Optional, List, Dict, Any

from openai import OpenAI

from agent.providers import BaseProvider, ChatResponse
from agent import config

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI GPT models."""

    name = "openai"

    def __init__(self, api_key: str | None = None, default_model: str | None = None):
        self._api_key = api_key or config.OPENAI_API_KEY
        self._default_model = default_model or config.OPENAI_GPT4O_MINI
        self._client: OpenAI | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> OpenAI:
        """Lazy-init the OpenAI client."""
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    @staticmethod
    def _parse_tool_calls(raw_tool_calls) -> Optional[List[Dict[str, Any]]]:
        """Convert SDK tool-call objects to plain dicts."""
        if not raw_tool_calls:
            return None
        parsed: list[dict] = []
        for tc in raw_tool_calls:
            parsed.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
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
        """Send a chat completion request and return a ChatResponse."""
        model = model or self._default_model
        client = self._get_client()

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")

        try:
            response = client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            logger.error("OpenAI API error: %s", exc)
            raise

        choice = response.choices[0]
        usage = response.usage

        return ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            provider=self.name,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            tool_calls=self._parse_tool_calls(choice.message.tool_calls),
            finish_reason=choice.finish_reason or "stop",
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
        """Stream tokens one at a time; return final ChatResponse."""
        model = model or self._default_model
        client = self._get_client()

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            **kwargs,
        }
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")

        try:
            stream = client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            logger.error("OpenAI streaming error: %s", exc)
            raise

        full_content: list[str] = []
        finish_reason = "stop"
        input_tokens = 0
        output_tokens = 0
        tool_call_chunks: dict[int, dict] = {}

        for chunk in stream:
            # Usage info arrives in the final chunk
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Accumulate streamed tool calls
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {
                            "id": tc_delta.id or "",
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

            # Yield text tokens
            if delta.content:
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
        """Return True if an OpenAI API key is configured."""
        return bool(self._api_key and self._api_key.strip())
