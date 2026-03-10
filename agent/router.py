"""
Smart Router v3 -- intent-based model routing with FREE_MODE safety lock.

Classifies user messages by intent (code, creative, reasoning, uncensored,
chat) and selects the optimal provider/model pair from the ranked
preference lists in config.INTENT_ROUTES.

When FREE_MODE=true (default), the router will NEVER route to a paid
provider -- guaranteed $0 spend. When FREE_MODE=false, paid providers
are allowed but still subject to daily/monthly budget caps.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Any, Optional

from agent import config
from agent.providers import BaseProvider
from agent.providers.universal_provider import UniversalProvider
from agent.budget import BudgetTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent keyword map -- order matters: first match wins
# ---------------------------------------------------------------------------
_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "creative": [
        "story", "poem", "creative", "imagine", "fiction",
        "blog", "essay", "compose", "write me",
    ],
    "code": [
        "code", "function", "debug", "python", "javascript", "program",
        "script", "implement", "class", "algorithm", "fix this", "write a",
        "refactor", "optimize", "compile", "syntax", "error", "bug",
        "api", "endpoint", "database", "sql", "html", "css", "react",
        "node", "typescript", "rust", "golang", "java", "c++",
    ],
    "reasoning": [
        "analyze", "compare", "reason", "think step by step",
        "pros and cons", "explain why", "evaluate", "assess",
        "break down", "logic", "proof", "derive", "calculate",
        "math", "solve", "figure out",
    ],
    "uncensored": [
        "uncensored", "no filter", "unfiltered", "no restrictions",
        "bypass", "jailbreak", "dolphin",
    ],
}


def _is_free_provider(prov_name: str) -> bool:
    """Check if a provider is in the free tier set."""
    return prov_name in config.FREE_PROVIDERS


def _model_has_cost(model: str) -> bool:
    """Check if a model has any non-zero cost."""
    cost = config.COST_PER_1K_TOKENS.get(model, {})
    return cost.get("input", 0) > 0 or cost.get("output", 0) > 0


def build_providers() -> Dict[str, UniversalProvider]:
    """
    Build all providers from config.PROVIDER_REGISTRY.

    Each entry in the registry becomes a UniversalProvider instance.
    Only providers with valid API keys (or local flag) are included.
    """
    providers: Dict[str, UniversalProvider] = {}

    for name, info in config.PROVIDER_REGISTRY.items():
        try:
            prov = UniversalProvider(
                name=name,
                api_key=info["api_key"],
                base_url=info["base_url"],
                default_model=info["default_model"],
                supports_tools=info.get("supports_tools", True),
                is_local=info.get("local", False),
            )
            providers[name] = prov

            status = "FREE" if info.get("free_tier") else "PAID"
            logger.info(
                "Registered provider: %s [%s] (available=%s)",
                name, status, prov.is_available(),
            )
        except Exception as e:
            logger.warning("Could not init provider %s: %s", name, e)

    if config.FREE_MODE:
        free_count = sum(
            1 for n, p in providers.items()
            if _is_free_provider(n) and p.is_available()
        )
        logger.info(
            "FREE_MODE=ON -- %d free providers available. "
            "Paid providers are registered but BLOCKED from routing.",
            free_count,
        )
    else:
        logger.warning(
            "FREE_MODE=OFF -- paid providers are UNLOCKED. "
            "Budget caps: $%.2f/day, $%.2f/month",
            config.DAILY_BUDGET_LIMIT,
            config.MONTHLY_BUDGET_LIMIT,
        )

    return providers


class Router:
    """Route user messages to the best available provider/model."""

    def __init__(
        self,
        providers: Dict[str, BaseProvider],
        budget: BudgetTracker,
    ):
        self.providers = providers
        self.budget = budget

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def classify_intent(self, message: str) -> str:
        """
        Classify a user message into an intent category.
        Scans the lowered message for keyword matches. First match wins.
        Default is "chat".
        """
        msg_lower = message.lower()
        for intent, keywords in _INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in msg_lower:
                    logger.debug("Intent '%s' matched keyword '%s'", intent, kw)
                    return intent
        return "chat"

    # ------------------------------------------------------------------
    # Provider eligibility check
    # ------------------------------------------------------------------

    def _is_eligible(self, prov_name: str, model: str, over_budget: bool) -> bool:
        """
        Check if a provider/model pair is eligible for routing.

        Rules (in order):
        1. FREE_MODE=true  -> BLOCK all paid providers, period.
        2. Over budget     -> BLOCK any model with cost > 0.
        3. Otherwise       -> Allow.
        """
        # Rule 1: FREE_MODE hard lock
        if config.FREE_MODE and not _is_free_provider(prov_name):
            return False

        # Rule 2: Budget exceeded -- only allow genuinely free models
        if over_budget and _model_has_cost(model):
            return False

        return True

    # ------------------------------------------------------------------
    # Main routing
    # ------------------------------------------------------------------

    def route(
        self,
        message: str,
    ) -> Tuple[BaseProvider, str, Dict[str, Any]]:
        """
        Select the best provider and model for a message.

        Logic:
        1. Classify intent via keyword matching.
        2. Walk the ranked preference list for that intent.
        3. Skip ineligible providers (FREE_MODE / budget checks).
        4. First available + eligible provider wins.
        5. If all intent providers fail, walk the global fallback chain.

        Returns (provider_instance, model_name, route_info_dict)
        """
        intent = self.classify_intent(message)
        over_budget = self.budget.is_over_budget()

        route_info: Dict[str, Any] = {
            "intent": intent,
            "provider": None,
            "model": None,
            "fallback": False,
            "budget_override": over_budget,
            "free_mode": config.FREE_MODE,
        }

        # Get the ranked provider list for this intent
        intent_chain = config.INTENT_ROUTES.get(intent, config.INTENT_ROUTES["chat"])

        # Walk the intent-specific chain
        for prov_name, model in intent_chain:
            if not self._is_eligible(prov_name, model, over_budget):
                continue

            provider = self.providers.get(prov_name)
            if provider and provider.is_available():
                route_info["provider"] = prov_name
                route_info["model"] = model
                logger.info(
                    "Routed: intent=%s -> %s/%s%s",
                    intent, prov_name, model,
                    " [FREE]" if _is_free_provider(prov_name) else " [PAID]",
                )
                return provider, model, route_info

        # Intent chain exhausted -- walk global fallback
        logger.warning(
            "All intent-specific providers unavailable for '%s', trying fallbacks...",
            intent,
        )

        for prov_name, model in config.FALLBACK_CHAIN:
            if not self._is_eligible(prov_name, model, over_budget):
                continue

            provider = self.providers.get(prov_name)
            if provider and provider.is_available():
                route_info["provider"] = prov_name
                route_info["model"] = model
                route_info["fallback"] = True
                logger.info(
                    "Fallback routed: %s/%s for intent '%s'%s",
                    prov_name, model, intent,
                    " [FREE]" if _is_free_provider(prov_name) else " [PAID]",
                )
                return provider, model, route_info

        # Nothing available at all
        if config.FREE_MODE:
            raise RuntimeError(
                "No FREE providers are available. Check API keys for: "
                + ", ".join(sorted(config.FREE_PROVIDERS))
                + "\nOr set FREE_MODE=false in .env to unlock paid providers."
            )
        else:
            raise RuntimeError(
                "No providers are available. Check API keys in .env "
                "and/or run 'ollama serve' for local models."
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

    def get_free_providers(self) -> List[str]:
        """Return names of available FREE providers only."""
        return [
            name
            for name, prov in self.providers.items()
            if prov.is_available() and _is_free_provider(name)
        ]

    def get_paid_providers(self) -> List[str]:
        """Return names of available PAID providers only."""
        return [
            name
            for name, prov in self.providers.items()
            if prov.is_available() and not _is_free_provider(name)
        ]

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Return all providers with availability and free/paid status."""
        return {
            name: {
                "available": prov.is_available(),
                "free": _is_free_provider(name),
                "blocked": config.FREE_MODE and not _is_free_provider(name),
                "limits": config.PROVIDER_REGISTRY.get(name, {}).get("limits", ""),
            }
            for name, prov in self.providers.items()
        }

    def __repr__(self) -> str:
        free = self.get_free_providers()
        paid = self.get_paid_providers()
        mode = "FREE_MODE" if config.FREE_MODE else "FULL_MODE"
        return (
            f"Router({mode}: {len(free)} free, {len(paid)} paid providers | "
            f"free=[{', '.join(free)}])"
        )
