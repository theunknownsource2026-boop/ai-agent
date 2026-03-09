"""
Smart Router — intent-based model routing with budget awareness.

Classifies user messages by intent (code, creative, reasoning, uncensored,
chat) and selects the optimal provider/model pair.  Falls back gracefully
when a provider is unavailable or the spending budget is exhausted.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Any, Optional

from agent import config
from agent.providers import BaseProvider
from agent.budget import BudgetTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent keyword map — order matters: first match wins
# ---------------------------------------------------------------------------
_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "creative": [
        "story", "poem", "creative", "imagine", "fiction",
        "blog", "essay", "compose",
    ],
    "code": [
        "code", "function", "debug", "python", "javascript", "program",
        "script", "implement", "class", "algorithm", "fix this", "write a",
    ],
    "reasoning": [
        "analyze", "compare", "reason", "think step by step",
        "pros and cons", "explain why", "evaluate", "assess",
    ],
    "uncensored": [
        "uncensored", "no filter", "unfiltered", "no restrictions",
        "bypass", "jailbreak", "dolphin",
    ],
}

# Intent -> (provider_key, model_constant)
_INTENT_ROUTES: Dict[str, Tuple[str, str]] = {
    "code":        ("mistral",  config.MISTRAL_CODESTRAL),
    "creative":    ("openai",   config.OPENAI_GPT4O),
    "reasoning":   ("openai",   config.OPENAI_GPT4O),
    "uncensored":  ("ollama",   config.OLLAMA_DOLPHIN),
    "chat":        ("groq",     config.GROQ_LLAMA_8B),
}

# Fallback order when the preferred provider is unavailable
_FALLBACK_ORDER: List[Tuple[str, str]] = [
    ("groq",    config.GROQ_LLAMA_8B),
    ("mistral", config.MISTRAL_SMALL),
    ("openai",  config.OPENAI_GPT4O_MINI),
    ("ollama",  config.OLLAMA_LLAMA),
]


class Router:
    """Route user messages to the best available provider/model."""

    def __init__(
        self,
        providers: Dict[str, BaseProvider],
        budget: BudgetTracker,
    ):
        """
        Parameters
        ----------
        providers : dict
            Mapping of provider name ("openai", "groq", ...) to a
            concrete :class:`BaseProvider` instance.
        budget : BudgetTracker
            Used to check spending limits before routing to paid APIs.
        """
        self.providers = providers
        self.budget = budget

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def classify_intent(self, message: str) -> Dict[str, str]:
        """
        Classify a user message into an intent category.

        Scans the lowered message for keyword matches.  The first
        intent category whose keyword appears wins.  When nothing
        matches the default is ``"chat"``.

        Returns
        -------
        dict
            ``{"intent": str, "provider": str, "model": str}``
        """
        msg_lower = message.lower()

        # Walk the keyword map; first matching intent wins
        for intent, keywords in _INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in msg_lower:
                    provider_key, model = _INTENT_ROUTES[intent]
                    logger.debug(
                        "Intent '%s' matched keyword '%s' -> %s/%s",
                        intent, kw, provider_key, model,
                    )
                    return {
                        "intent": intent,
                        "provider": provider_key,
                        "model": model,
                    }

        # Default fallback: chat
        provider_key, model = _INTENT_ROUTES["chat"]
        return {"intent": "chat", "provider": provider_key, "model": model}

    # ------------------------------------------------------------------
    # Main routing
    # ------------------------------------------------------------------

    def route(
        self,
        message: str,
    ) -> Tuple[BaseProvider, str, Dict[str, Any]]:
        """
        Select the best provider and model for *message*.

        Logic:
        1. Classify the intent via keyword matching.
        2. If the budget is exceeded, override to ollama/llama3.1 (free).
        3. Check that the chosen provider is available.
        4. If not, walk the fallback chain: groq -> mistral -> openai -> ollama.

        Returns
        -------
        tuple
            ``(provider_instance, model_name, route_info_dict)``

        Raises
        ------
        RuntimeError
            If no provider is available at all.
        """
        route_info = self.classify_intent(message)
        intent = route_info["intent"]
        provider_key = route_info["provider"]
        model = route_info["model"]

        # ---- Budget guard: force free local model when over budget ----
        if self.budget.is_over_budget():
            logger.warning(
                "Budget exceeded — forcing ollama/%s for intent '%s'",
                config.OLLAMA_LLAMA, intent,
            )
            provider_key = "ollama"
            model = config.OLLAMA_LLAMA
            route_info.update(
                provider=provider_key,
                model=model,
                budget_override=True,
            )

        # ---- Availability check with fallback chain -----------------
        provider = self.providers.get(provider_key)

        if provider and provider.is_available():
            route_info["fallback"] = False
            logger.info(
                "Routed: intent=%s -> %s/%s", intent, provider_key, model,
            )
            return provider, model, route_info

        # Preferred provider not available — walk the fallback list
        logger.warning(
            "Provider '%s' unavailable, searching fallbacks...", provider_key,
        )
        for fb_key, fb_model in _FALLBACK_ORDER:
            if fb_key == provider_key:
                continue  # already tried
            fb_provider = self.providers.get(fb_key)
            if fb_provider and fb_provider.is_available():
                logger.info(
                    "Fallback: %s/%s -> %s/%s",
                    provider_key, model, fb_key, fb_model,
                )
                route_info.update(
                    provider=fb_key,
                    model=fb_model,
                    fallback=True,
                    original_provider=provider_key,
                    original_model=model,
                )
                return fb_provider, fb_model, route_info

        raise RuntimeError(
            "No providers are available. Check API keys and Ollama status."
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_available_providers(self) -> List[str]:
        """Return the names of providers that are currently reachable."""
        return [
            name
            for name, prov in self.providers.items()
            if prov.is_available()
        ]

    def __repr__(self) -> str:
        avail = self.get_available_providers()
        return f"Router(providers={list(self.providers.keys())}, available={avail})"
