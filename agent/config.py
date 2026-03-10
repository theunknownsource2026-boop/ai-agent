"""
Agent configuration module.

Loads environment variables, defines model constants, pricing,
budget limits, and the default system prompt.

Supports 14 providers -- prioritizes 100% free tiers to avoid spending.

FREE TIER REALITY (verified March 2026):
  Groq        -- 100% free, 30 RPM, ~1K req/day
  Gemini      -- 100% free, 15 RPM, 250K TPM
  Cerebras    -- 100% free tier, fastest inference
  OpenRouter  -- 100% free on :free models, 20 RPM, 50 req/day
  Ollama      -- 100% free (local hardware)
  Mistral     -- 100% free, 1 RPS, 500K TPM, ~1B tokens/mo
  Cohere      -- Free trial, 1000 calls/mo, 20 RPM (non-commercial)
  DeepSeek    -- Near-free, $0.14/$0.28 per 1M tokens
  xAI         -- $25 signup credit + $150/mo with data sharing opt-in
  Anthropic   -- $5 signup credit then paid
  OpenAI      -- 3 RPM free tier (very limited), then pay-as-you-go
  Together AI -- NO free tier, $5 minimum purchase
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root (two levels up from this file, or cwd)
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()  # fall back to cwd / parent search

# ---------------------------------------------------------------------------
# API Keys  (11 providers you have keys for + ollama local)
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
CEREBRAS_API_KEY: str = os.getenv("CEREBRAS_API_KEY", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
TOGETHER_API_KEY: str = os.getenv("TOGETHER_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# FREE_MODE -- when True, router will ONLY use providers with $0.00 cost
# Set to "false" in .env to unlock paid providers when you want to
# ---------------------------------------------------------------------------
FREE_MODE: bool = os.getenv("FREE_MODE", "true").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Model name constants
# ---------------------------------------------------------------------------

# --- 100% FREE PROVIDERS (no spend ever) ---

# Groq (free, fastest open-source inference)
GROQ_LLAMA_70B = "llama-3.3-70b-versatile"
GROQ_LLAMA_8B = "llama-3.1-8b-instant"
GROQ_QWEN3 = "qwen-qwq-32b"

# Google Gemini (free tier via OpenAI-compatible endpoint)
GEMINI_PRO = "gemini-2.5-pro-preview-06-05"
GEMINI_FLASH = "gemini-2.0-flash"
GEMINI_FLASH_LITE = "gemini-2.0-flash-lite"

# Cerebras (free, ultra-fast inference)
CEREBRAS_LLAMA_70B = "llama-3.3-70b"
CEREBRAS_QWEN = "qwen-3-32b"

# OpenRouter (free models only -- must use :free suffix)
OPENROUTER_AUTO_FREE = "openrouter/auto"      # auto-selects best free model
OPENROUTER_DEEPSEEK_FREE = "deepseek/deepseek-r1:free"
OPENROUTER_LLAMA_FREE = "meta-llama/llama-4-scout:free"

# Ollama (local, unlimited, $0)
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "dolphin-llama3")
OLLAMA_DOLPHIN = "dolphin-llama3"
OLLAMA_LLAMA = "llama3.1"

# Mistral (free tier: 1 RPS, 500K TPM, ~1B tokens/mo)
MISTRAL_SMALL = "mistral-small-latest"
MISTRAL_CODESTRAL = "codestral-latest"
MISTRAL_LARGE = "mistral-large-latest"

# Cohere (free trial: 1000 calls/mo, 20 RPM -- non-commercial)
COHERE_COMMAND_R = "command-r"
COHERE_COMMAND_R_PLUS = "command-r-plus"

# --- CREDIT-BASED PROVIDERS (free until credits run out) ---

# DeepSeek (near-free: $0.14 input / $0.28 output per 1M tokens)
DEEPSEEK_CHAT = "deepseek-chat"
DEEPSEEK_REASONER = "deepseek-reasoner"

# xAI / Grok ($25 signup credit; $150/mo if you opt into data sharing)
XAI_GROK_FAST = "grok-4.1-fast"       # $0.20/$0.50 per 1M -- best value
XAI_GROK_MINI = "grok-3-mini"         # $0.10/$0.30 per 1M -- cheapest

# Anthropic ($5 signup credit then paid)
ANTHROPIC_HAIKU = "claude-haiku-4.5"
ANTHROPIC_SONNET = "claude-sonnet-4.6"

# OpenAI (3 RPM free tier on mini, then pay-as-you-go)
OPENAI_GPT4O_MINI = "gpt-4o-mini"     # $0.15/$0.60 per 1M
OPENAI_GPT4O = "gpt-4o"               # $2.50/$10.00 per 1M

# Together AI (NO free tier -- $5 minimum purchase)
TOGETHER_LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"  # $0.06/1M cheapest
TOGETHER_LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct"  # $0.88/1M


# ---------------------------------------------------------------------------
# Provider registry -- everything the router needs to know
# "free_tier": True means this provider costs $0 within its limits
# ---------------------------------------------------------------------------
PROVIDER_REGISTRY: dict[str, dict] = {
    # === 100% FREE PROVIDERS (ordered by quality) ===
    "groq": {
        "api_key": GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": GROQ_LLAMA_70B,
        "supports_tools": True,
        "models": [GROQ_LLAMA_70B, GROQ_LLAMA_8B, GROQ_QWEN3],
        "free_tier": True,
        "limits": "30 RPM, ~1K req/day",
    },
    "gemini": {
        "api_key": GOOGLE_API_KEY,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": GEMINI_FLASH,
        "supports_tools": True,
        "models": [GEMINI_PRO, GEMINI_FLASH, GEMINI_FLASH_LITE],
        "free_tier": True,
        "limits": "15 RPM, 250K TPM",
    },
    "cerebras": {
        "api_key": CEREBRAS_API_KEY,
        "base_url": "https://api.cerebras.ai/v1",
        "default_model": CEREBRAS_LLAMA_70B,
        "supports_tools": True,
        "models": [CEREBRAS_LLAMA_70B, CEREBRAS_QWEN],
        "free_tier": True,
        "limits": "Free tier, fastest inference",
    },
    "openrouter": {
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": OPENROUTER_AUTO_FREE,
        "supports_tools": True,
        "models": [OPENROUTER_AUTO_FREE, OPENROUTER_DEEPSEEK_FREE, OPENROUTER_LLAMA_FREE],
        "free_tier": True,
        "limits": "20 RPM, 50 req/day (free models only)",
    },
    "ollama": {
        "api_key": "ollama",  # ollama doesn't need a real key
        "base_url": f"{OLLAMA_BASE_URL}/v1",
        "default_model": OLLAMA_LLAMA,
        "supports_tools": False,
        "models": [OLLAMA_DOLPHIN, OLLAMA_LLAMA],
        "free_tier": True,
        "local": True,
        "limits": "Unlimited (local hardware)",
    },
    "mistral": {
        "api_key": MISTRAL_API_KEY,
        "base_url": "https://api.mistral.ai/v1",
        "default_model": MISTRAL_SMALL,
        "supports_tools": True,
        "models": [MISTRAL_LARGE, MISTRAL_SMALL, MISTRAL_CODESTRAL],
        "free_tier": True,
        "limits": "1 RPS, 500K TPM, ~1B tokens/mo",
    },
    "cohere": {
        "api_key": COHERE_API_KEY,
        "base_url": "https://api.cohere.com/compatibility/v1",
        "default_model": COHERE_COMMAND_R,
        "supports_tools": True,
        "models": [COHERE_COMMAND_R, COHERE_COMMAND_R_PLUS],
        "free_tier": True,
        "limits": "1000 calls/mo, 20 RPM (trial key, non-commercial)",
    },
    # === CREDIT-BASED / PAID PROVIDERS ===
    "deepseek": {
        "api_key": DEEPSEEK_API_KEY,
        "base_url": "https://api.deepseek.com/v1",
        "default_model": DEEPSEEK_CHAT,
        "supports_tools": True,
        "models": [DEEPSEEK_CHAT, DEEPSEEK_REASONER],
        "free_tier": False,
        "limits": "Pay-as-you-go, ~$0.14/$0.28 per 1M tokens (extremely cheap)",
    },
    "xai": {
        "api_key": XAI_API_KEY,
        "base_url": "https://api.x.ai/v1",
        "default_model": XAI_GROK_MINI,
        "supports_tools": True,
        "models": [XAI_GROK_FAST, XAI_GROK_MINI],
        "free_tier": False,
        "limits": "$25 signup credit; $150/mo with data sharing opt-in",
    },
    "anthropic": {
        "api_key": ANTHROPIC_API_KEY,
        "base_url": "https://api.anthropic.com/v1",
        "default_model": ANTHROPIC_HAIKU,
        "supports_tools": True,
        "models": [ANTHROPIC_HAIKU, ANTHROPIC_SONNET],
        "free_tier": False,
        "limits": "$5 signup credit then paid; Haiku $1/$5 per 1M",
    },
    "openai": {
        "api_key": OPENAI_API_KEY,
        "base_url": "https://api.openai.com/v1",
        "default_model": OPENAI_GPT4O_MINI,
        "supports_tools": True,
        "models": [OPENAI_GPT4O, OPENAI_GPT4O_MINI],
        "free_tier": False,
        "limits": "3 RPM free tier (limited), then $5 minimum top-up",
    },
    "together": {
        "api_key": TOGETHER_API_KEY,
        "base_url": "https://api.together.xyz/v1",
        "default_model": TOGETHER_LLAMA_3B,
        "supports_tools": True,
        "models": [TOGETHER_LLAMA_3B, TOGETHER_LLAMA_70B],
        "free_tier": False,
        "limits": "NO free tier -- $5 minimum purchase required",
    },
}

# ---------------------------------------------------------------------------
# Sets for quick lookups in router
# ---------------------------------------------------------------------------
FREE_PROVIDERS: set[str] = {
    name for name, info in PROVIDER_REGISTRY.items()
    if info.get("free_tier", False)
}
# {"groq", "gemini", "cerebras", "openrouter", "ollama", "mistral", "cohere"}

PAID_PROVIDERS: set[str] = {
    name for name, info in PROVIDER_REGISTRY.items()
    if not info.get("free_tier", False)
}
# {"deepseek", "xai", "anthropic", "openai", "together"}


# ---------------------------------------------------------------------------
# Cost per 1,000 tokens (USD)
# 0.0 = genuinely free within provider limits
# ---------------------------------------------------------------------------
COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    # === FREE ($0.00) ===
    # Groq
    GROQ_LLAMA_70B:          {"input": 0.0,      "output": 0.0},
    GROQ_LLAMA_8B:           {"input": 0.0,      "output": 0.0},
    GROQ_QWEN3:              {"input": 0.0,      "output": 0.0},
    # Gemini
    GEMINI_PRO:              {"input": 0.0,      "output": 0.0},
    GEMINI_FLASH:            {"input": 0.0,      "output": 0.0},
    GEMINI_FLASH_LITE:       {"input": 0.0,      "output": 0.0},
    # Cerebras
    CEREBRAS_LLAMA_70B:      {"input": 0.0,      "output": 0.0},
    CEREBRAS_QWEN:           {"input": 0.0,      "output": 0.0},
    # OpenRouter free models
    OPENROUTER_AUTO_FREE:    {"input": 0.0,      "output": 0.0},
    OPENROUTER_DEEPSEEK_FREE: {"input": 0.0,     "output": 0.0},
    OPENROUTER_LLAMA_FREE:   {"input": 0.0,      "output": 0.0},
    # Ollama (local)
    OLLAMA_DOLPHIN:          {"input": 0.0,      "output": 0.0},
    OLLAMA_LLAMA:            {"input": 0.0,      "output": 0.0},
    # Mistral (free tier)
    MISTRAL_LARGE:           {"input": 0.0,      "output": 0.0},
    MISTRAL_SMALL:           {"input": 0.0,      "output": 0.0},
    MISTRAL_CODESTRAL:       {"input": 0.0,      "output": 0.0},
    # Cohere (free trial)
    COHERE_COMMAND_R:        {"input": 0.0,      "output": 0.0},
    COHERE_COMMAND_R_PLUS:   {"input": 0.0,      "output": 0.0},

    # === PAID (uses credits or real money) ===
    # DeepSeek (near-free)
    DEEPSEEK_CHAT:           {"input": 0.00014,  "output": 0.00028},
    DEEPSEEK_REASONER:       {"input": 0.00055,  "output": 0.0022},
    # xAI / Grok
    XAI_GROK_FAST:           {"input": 0.0002,   "output": 0.0005},
    XAI_GROK_MINI:           {"input": 0.0001,   "output": 0.0003},
    # Anthropic
    ANTHROPIC_HAIKU:         {"input": 0.001,    "output": 0.005},
    ANTHROPIC_SONNET:        {"input": 0.003,    "output": 0.015},
    # OpenAI
    OPENAI_GPT4O_MINI:       {"input": 0.00015,  "output": 0.0006},
    OPENAI_GPT4O:            {"input": 0.0025,   "output": 0.01},
    # Together AI
    TOGETHER_LLAMA_3B:       {"input": 0.00006,  "output": 0.00006},
    TOGETHER_LLAMA_70B:      {"input": 0.00088,  "output": 0.00088},
}

# ---------------------------------------------------------------------------
# Intent -> provider routing
# FREE_MODE=true: only the free entries are used (router enforces this)
# FREE_MODE=false: full chain including paid providers
# ---------------------------------------------------------------------------
INTENT_ROUTES: dict[str, list[tuple[str, str]]] = {
    # coding: best free code models first, paid fallbacks last
    "code": [
        ("mistral", MISTRAL_CODESTRAL),       # free -- purpose-built for code
        ("groq", GROQ_LLAMA_70B),             # free -- fast
        ("cerebras", CEREBRAS_LLAMA_70B),      # free -- ultra-fast
        ("gemini", GEMINI_FLASH),              # free
        ("openrouter", OPENROUTER_DEEPSEEK_FREE),  # free
        # --- paid fallbacks (blocked when FREE_MODE=true) ---
        ("deepseek", DEEPSEEK_CHAT),           # $0.14/1M -- near free
        ("xai", XAI_GROK_FAST),               # credits
        ("anthropic", ANTHROPIC_HAIKU),        # credits
        ("openai", OPENAI_GPT4O_MINI),         # paid
    ],
    # reasoning: strong reasoners first
    "reasoning": [
        ("groq", GROQ_QWEN3),                 # free -- QwQ reasoning model
        ("gemini", GEMINI_PRO),                # free -- very strong
        ("cerebras", CEREBRAS_QWEN),           # free
        ("openrouter", OPENROUTER_DEEPSEEK_FREE),  # free -- DeepSeek R1
        ("mistral", MISTRAL_LARGE),            # free tier
        # --- paid fallbacks ---
        ("deepseek", DEEPSEEK_REASONER),       # $0.55/1M -- purpose-built
        ("anthropic", ANTHROPIC_SONNET),       # credits
        ("openai", OPENAI_GPT4O),              # paid
    ],
    # creative writing
    "creative": [
        ("gemini", GEMINI_PRO),                # free -- great creative
        ("groq", GROQ_LLAMA_70B),             # free
        ("cohere", COHERE_COMMAND_R_PLUS),     # free trial
        ("mistral", MISTRAL_LARGE),            # free tier
        ("cerebras", CEREBRAS_LLAMA_70B),      # free
        # --- paid fallbacks ---
        ("anthropic", ANTHROPIC_SONNET),       # credits -- best creative
        ("xai", XAI_GROK_FAST),               # credits
        ("openai", OPENAI_GPT4O),              # paid
    ],
    # general chat: fastest free providers first
    "chat": [
        ("groq", GROQ_LLAMA_70B),             # free -- fastest
        ("cerebras", CEREBRAS_LLAMA_70B),      # free -- ultra-fast
        ("gemini", GEMINI_FLASH),              # free
        ("mistral", MISTRAL_SMALL),            # free tier
        ("openrouter", OPENROUTER_AUTO_FREE),  # free
        ("cohere", COHERE_COMMAND_R),          # free trial
        # --- paid fallbacks ---
        ("deepseek", DEEPSEEK_CHAT),           # near-free
        ("xai", XAI_GROK_MINI),               # credits
        ("openai", OPENAI_GPT4O_MINI),         # paid
        ("together", TOGETHER_LLAMA_3B),       # paid
    ],
    # uncensored: local ollama only
    "uncensored": [
        ("ollama", OLLAMA_DOLPHIN),
    ],
}

# Global fallback chain -- all free providers first, paid absolute last resort
FALLBACK_CHAIN: list[tuple[str, str]] = [
    # --- FREE (try all of these first) ---
    ("groq", GROQ_LLAMA_8B),              # free -- lightest/fastest
    ("cerebras", CEREBRAS_LLAMA_70B),      # free
    ("gemini", GEMINI_FLASH_LITE),         # free -- lightest Gemini
    ("openrouter", OPENROUTER_LLAMA_FREE), # free
    ("mistral", MISTRAL_SMALL),            # free tier
    ("cohere", COHERE_COMMAND_R),          # free trial
    ("ollama", OLLAMA_LLAMA),              # free local
    # --- PAID (last resort, blocked in FREE_MODE) ---
    ("deepseek", DEEPSEEK_CHAT),           # near-free
    ("xai", XAI_GROK_MINI),               # credits
    ("openai", OPENAI_GPT4O_MINI),         # paid
    ("anthropic", ANTHROPIC_HAIKU),        # credits
    ("together", TOGETHER_LLAMA_3B),       # paid
]

# ---------------------------------------------------------------------------
# Budget limits -- safety net even if FREE_MODE is off
# Set low so you never accidentally blow money
# ---------------------------------------------------------------------------
DAILY_BUDGET_LIMIT: float = float(os.getenv("DAILY_BUDGET_LIMIT", "0.50"))
MONTHLY_BUDGET_LIMIT: float = float(os.getenv("MONTHLY_BUDGET_LIMIT", "5.00"))

# ---------------------------------------------------------------------------
# Provider-specific credit budgets (track per-provider spend)
# Set to match each provider's actual free credit amount
# ---------------------------------------------------------------------------
PROVIDER_CREDIT_LIMITS: dict[str, float] = {
    "anthropic": 5.00,     # $5 signup credit
    "xai": 25.00,          # $25 signup credit (not counting data-sharing $150/mo)
    "openai": 0.00,        # no free credits in 2026
    "together": 0.00,      # no free tier at all
    "deepseek": 0.00,      # no free credits, just cheap pricing
    # Free providers don't need credit tracking
}

# ---------------------------------------------------------------------------
# Database path (for persistent memory + chat history)
# ---------------------------------------------------------------------------
DB_PATH: str = os.getenv("AGENT_DB_PATH", str(_project_root / "agent_data.db"))

# ---------------------------------------------------------------------------
# Knowledge folder (auto-ingest documents on startup)
# ---------------------------------------------------------------------------
KNOWLEDGE_DIR: str = os.getenv("KNOWLEDGE_DIR", str(_project_root / "knowledge"))
AUTO_INGEST_DIR: str = os.getenv("AUTO_INGEST_DIR", str(_project_root / "knowledge"))

# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT: str = (
    "You are a helpful AI assistant. Be concise and direct in your responses. "
    "Use tools when they would provide better answers than your training data alone. "
    "If you are unsure about something, say so rather than guessing. "
    "When presenting information, structure it clearly with bullet points or "
    "numbered lists when appropriate."
)
