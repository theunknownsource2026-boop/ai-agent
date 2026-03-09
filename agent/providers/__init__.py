"""
Provider base classes and unified ChatResponse dataclass.

Every provider (OpenAI, Groq, Mistral, Ollama) subclasses BaseProvider
so the rest of the codebase never cares which backend is in use.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Generator
from abc import ABC, abstractmethod


@dataclass
class ChatResponse:
    """Unified response from any provider."""

    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"

    # ------ convenience helpers ------

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed by this request."""
        return self.input_tokens + self.output_tokens

    @property
    def has_tool_calls(self) -> bool:
        """True when the model wants to invoke one or more tools."""
        return bool(self.tool_calls)

    def __str__(self) -> str:
        tc = f"  tool_calls={len(self.tool_calls)}" if self.tool_calls else ""
        return (
            f"ChatResponse(provider={self.provider}, model={self.model}, "
            f"tokens={self.total_tokens}{tc})"
        )


class BaseProvider(ABC):
    """All providers implement this interface."""

    name: str = "base"

    @abstractmethod
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
        """Send messages, get a ChatResponse back."""
        pass

    @abstractmethod
    def stream_chat(
        self,
        messages: list,
        model: str = None,
        tools: list = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> Generator[str, None, ChatResponse]:
        """Yield tokens one at a time, return final ChatResponse."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and reachable."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} available={self.is_available()}>"
