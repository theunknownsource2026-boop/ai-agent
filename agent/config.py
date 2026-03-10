"""
Agent configuration module.

Loads environment variables, defines model constants, pricing,
budget limits, and the default system prompt.

Supports 12+ providers -- most are OpenAI-compatible endpoints.
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
# API Keys
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
CEREBRAS_API_KEY: str = os.getenv("CEREBRAS_API_KEY", "")
GITHUB_MODELS_TOKEN: str = os.getenv("GITHUB_MODELS_TOKEN", "")
SAMBANOVA_API_KEY: str = os.getenv("SAMBANOVA_API_KEY", "")
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")

# ---------------------------------------------------------------------------
# Model name constants
# ---------------------------------------------------------------------------

# OpenAI
OPENAI_GPT4O = "gpt-4o"
OPENAI_GPT4O_MINI = "gpt-4o-mini"

# Groq (hosted open-source, fastest inference)
GROQ_LLAMA_70B = "llama-3.3-70b-versatile"
GROQ_LLAMA_8B = "llama-3.1-8b-instant"
GROQ_QWEN3 = "qwen-qwq-32b"

# Mistral
MISTRAL_LARGE = "mistral-large-latest"
MISTRAL_SMALL = "mistral-small-latest"
MISTRAL_CODESTRAL = "codestral-latest"

# Ollama (local)
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "dolphin-llama3")
OLLAMA_DOLPHIN = "dolphin-llama3"
OLLAMA_LLAMA = "llama3.1"

# Google Gemini (via OpenAI-compatible endpoint)
GEMINI_PRO = "gemini-2.5-pro-preview-06-05"
GEMINI_FLASH = "gemini-2.0-flash"
GEMINI_FLASH_LITE = "gemini-2.0-flash-lite"

# DeepSeek
DEEPSEEK_CHAT = "deepseek-chat"
DEEPSEEK_REASONER = "deepseek-reasoner"

# OpenRouter (gateway to 100+ models)
OPENROUTER_FREE = "openrouter/auto"  # auto-selects best free model
OPENROUTER_DEEPSEEK = "deepseek/deepseek-r1:free"
OPENROUTER_LLAMA = "meta-llama/llama-4-scout:free"

# Cerebras (ultra-fast inference)
CEREBRAS_LLAMA_70B = "llama-3.3-70b"
CEREBRAS_QWEN = "qwen-3-32b"

# GitHub Models
GITHUB_GPT4O = "gpt-4o"
GITHUB_GPT4O_MINI = "gpt-4o-mini"

# SambaNova
SAMBANOVA_LLAMA_70B = "Meta-Llama-3.3-70B-Instruct"

# NVIDIA NIM
NVIDIA_LLAMA_70B = "meta/llama-3.3-70b-instruct"

# Cohere
COHERE_COMMAND_R_PLUS = "command-r-plus"

# ---------------------------------------------------------------------------
# Provider registry -- everything the router needs to know
# Each entry: (api_key_env, base_url, default_model, supports_tools)
# ---------------------------------------------------------------------------
PROVIDER_REGISTRY: dict[str, dict] = {
    "openai": {
        "api_key": OPENAI_API_KEY,
        "base_url": "https://api.openai.com/v1",
        "default_model": OPENAI_GPT4O_MINI,
        "supports_tools": True,
        "models": [OPENAI_GPT4O, OPENAI_GPT4O_MINI],
    },
    "groq": {
        "api_key": GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": GROQ_LLAMA_70B,
        "supports_tools": True,
        "models": [GROQ_LLAMA_70B, GROQ_LLAMA_8B, GROQ_QWEN3],
    },
    "mistral": {
        "api_key": MISTRAL_API_KEY,
        "base_url": "https://api.mistral.ai/v1",
        "default_model": MISTRAL_SMALL,
        "supports_tools": True,
        "models": [MISTRAL_LARGE, MISTRAL_SMALL, MISTRAL_CODESTRAL],
    },
    "ollama": {
        "api_key": "ollama",  # ollama doesn't need a real key
        "base_url": f"{OLLAMA_BASE_URL}/v1",
        "default_model": OLLAMA_LLAMA,
        "supports_tools": False,
        "models": [OLLAMA_DOLPHIN, OLLAMA_LLAMA],
        "local": True,
    },
    "gemini": {
        "api_key": GOOGLE_API_KEY,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": GEMINI_FLASH,
        "supports_tools": True,
        "models": [GEMINI_PRO, GEMINI_FLASH, GEMINI_FLASH_LITE],
    },
    "deepseek": {
        "api_key": DEEPSEEK_API_KEY,
        "base_url": "https://api.deepseek.com/v1",
        "default_model": DEEPSEEK_CHAT,
        "supports_tools": True,
        "models": [DEEPSEEK_CHAT, DEEPSEEK_REASONER],
    },
    "openrouter": {
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": OPENROUTER_FREE,
        "supports_tools": True,
        "models": [OPENROUTER_FREE, OPENROUTER_DEEPSEEK, OPENROUTER_LLAMA],
    },
    "cerebras": {
        "api_key": CEREBRAS_API_KEY,
        "base_url": "https://api.cerebras.ai/v1",
        "default_model": CEREBRAS_LLAMA_70B,
        "supports_tools": True,
        "models": [CEREBRAS_LLAMA_70B, CEREBRAS_QWEN],
    },
    "github": {
        "api_key": GITHUB_MODELS_TOKEN,
        "base_url": "https://models.inference.ai.azure.com",
        "default_model": GITHUB_GPT4O_MINI,
        "supports_tools": True,
        "models": [GITHUB_GPT4O, GITHUB_GPT4O_MINI],
    },
    "sambanova": {
        "api_key": SAMBANOVA_API_KEY,
        "base_url": "https://api.sambanova.ai/v1",
        "default_model": SAMBANOVA_LLAMA_70B,
        "supports_tools": True,
        "models": [SAMBANOVA_LLAMA_70B],
    },
    "nvidia": {
        "api_key": NVIDIA_API_KEY,
        "base_url": "https://integrate.api.nvidia.com/v1",
        "default_model": NVIDIA_LLAMA_70B,
        "supports_tools": True,
        "models": [NVIDIA_LLAMA_70B],
    },
    "cohere": {
        "api_key": COHERE_API_KEY,
        "base_url": "https://api.cohere.com/compatibility/v1",
        "default_model": COHERE_COMMAND_R_PLUS,
        "supports_tools": True,
        "models": [COHERE_COMMAND_R_PLUS],
    },
}

# ---------------------------------------------------------------------------
# Cost per 1 000 tokens (USD) -- 0.0 = free tier
# ---------------------------------------------------------------------------
COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    # OpenAI
    OPENAI_GPT4O:       {"input": 0.0025,  "output": 0.01},
    OPENAI_GPT4O_MINI:  {"input": 0.00015, "output": 0.0006},
    # Groq (free)
    GROQ_LLAMA_70B:     {"input": 0.0, "output": 0.0},
    GROQ_LLAMA_8B:      {"input": 0.0, "output": 0.0},
    GROQ_QWEN3:         {"input": 0.0, "output": 0.0},
    # Mistral (free tier)
    MISTRAL_LARGE:      {"input": 0.002,   "output": 0.006},
    MISTRAL_SMALL:      {"input": 0.0002,  "output": 0.0006},
    MISTRAL_CODESTRAL:  {"input": 0.0003,  "output": 0.0009},
    # Ollama (free / local)
    OLLAMA_DOLPHIN:     {"input": 0.0, "output": 0.0},
    OLLAMA_LLAMA:       {"input": 0.0, "output": 0.0},
    # Gemini (free tier)
    GEMINI_PRO:         {"input": 0.0, "output": 0.0},
    GEMINI_FLASH:       {"input": 0.0, "output": 0.0},
    GEMINI_FLASH_LITE:  {"input": 0.0, "output": 0.0},
    # DeepSeek (near-free)
    DEEPSEEK_CHAT:      {"input": 0.00014, "output": 0.00028},
    DEEPSEEK_REASONER:  {"input": 0.00055, "output": 0.0022},
    # OpenRouter free models
    OPENROUTER_FREE:    {"input": 0.0, "output": 0.0},
    OPENROUTER_DEEPSEEK: {"input": 0.0, "output": 0.0},
    OPENROUTER_LLAMA:   {"input": 0.0, "output": 0.0},
    # Cerebras (free)
    CEREBRAS_LLAMA_70B: {"input": 0.0, "output": 0.0},
    CEREBRAS_QWEN:      {"input": 0.0, "output": 0.0},
    # GitHub Models (free)
    GITHUB_GPT4O:       {"input": 0.0, "output": 0.0},
    GITHUB_GPT4O_MINI:  {"input": 0.0, "output": 0.0},
    # SambaNova (free)
    SAMBANOVA_LLAMA_70B: {"input": 0.0, "output": 0.0},
    # NVIDIA (free credits)
    NVIDIA_LLAMA_70B:   {"input": 0.0, "output": 0.0},
    # Cohere (free tier)
    COHERE_COMMAND_R_PLUS: {"input": 0.0, "output": 0.0},
}

# ---------------------------------------------------------------------------
# Intent -> provider routing (which provider is BEST for each task type)
# ---------------------------------------------------------------------------
INTENT_ROUTES: dict[str, list[tuple[str, str]]] = {
    # coding: DeepSeek > Mistral Codestral > Groq > Gemini
    "code": [
        ("deepseek", DEEPSEEK_CHAT),
        ("mistral", MISTRAL_CODESTRAL),
        ("groq", GROQ_LLAMA_70B),
        ("gemini", GEMINI_FLASH),
        ("cerebras", CEREBRAS_LLAMA_70B),
    ],
    # reasoning: DeepSeek R1 > Gemini Pro > OpenAI GPT-4o
    "reasoning": [
        ("deepseek", DEEPSEEK_REASONER),
        ("gemini", GEMINI_PRO),
        ("openai", OPENAI_GPT4O),
        ("groq", GROQ_LLAMA_70B),
    ],
    # creative writing: Gemini > OpenAI > Groq
    "creative": [
        ("gemini", GEMINI_PRO),
        ("openai", OPENAI_GPT4O),
        ("groq", GROQ_LLAMA_70B),
        ("cohere", COHERE_COMMAND_R_PLUS),
    ],
    # general chat: Groq (fastest) > Cerebras > Gemini Flash
    "chat": [
        ("groq", GROQ_LLAMA_70B),
        ("cerebras", CEREBRAS_LLAMA_70B),
        ("gemini", GEMINI_FLASH),
        ("openrouter", OPENROUTER_FREE),
        ("sambanova", SAMBANOVA_LLAMA_70B),
    ],
    # uncensored: local ollama only
    "uncensored": [
        ("ollama", OLLAMA_DOLPHIN),
    ],
}

# Global fallback chain if everything in INTENT_ROUTES fails
FALLBACK_CHAIN: list[tuple[str, str]] = [
    ("groq", GROQ_LLAMA_8B),
    ("gemini", GEMINI_FLASH_LITE),
    ("cerebras", CEREBRAS_LLAMA_70B),
    ("openrouter", OPENROUTER_FREE),
    ("sambanova", SAMBANOVA_LLAMA_70B),
    ("mistral", MISTRAL_SMALL),
    ("openai", OPENAI_GPT4O_MINI),
    ("ollama", OLLAMA_LLAMA),
]

# ---------------------------------------------------------------------------
# Budget limits
# ---------------------------------------------------------------------------
DAILY_BUDGET_LIMIT: float = float(os.getenv("DAILY_BUDGET_LIMIT", "2.00"))
MONTHLY_BUDGET_LIMIT: float = float(os.getenv("MONTHLY_BUDGET_LIMIT", "30.00"))

# ---------------------------------------------------------------------------
# Auto-ingest directory (for auto-learning from docs)
# ---------------------------------------------------------------------------
AUTO_INGEST_DIR: str = os.getenv("AUTO_INGEST_DIR", str(_project_root / "knowledge"))

# ---------------------------------------------------------------------------
# Database path (for persistent memory + chat history)
# ---------------------------------------------------------------------------
DB_PATH: str = os.getenv("AGENT_DB_PATH", str(_project_root / "agent_data.db"))

# ---------------------------------------------------------------------------
# Knowledge folder (auto-ingest documents on startup)
# ---------------------------------------------------------------------------
KNOWLEDGE_DIR: str = os.getenv("KNOWLEDGE_DIR", str(_project_root / "knowledge"))

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
