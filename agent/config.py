"""
Agent configuration module.

Loads environment variables, defines model constants, pricing,
budget limits, and the default system prompt.
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

# ---------------------------------------------------------------------------
# Model name constants  (provider -> list of supported model strings)
# ---------------------------------------------------------------------------

# OpenAI
OPENAI_GPT4O = "gpt-4o"
OPENAI_GPT4O_MINI = "gpt-4o-mini"

# Groq  (hosted open-source models)
GROQ_LLAMA_70B = "llama-3.1-70b-versatile"
GROQ_LLAMA_8B = "llama-3.1-8b-instant"

# Mistral
MISTRAL_LARGE = "mistral-large-latest"
MISTRAL_SMALL = "mistral-small-latest"
MISTRAL_CODESTRAL = "codestral-latest"

# Ollama  (local models)
OLLAMA_DOLPHIN = "dolphin-mistral"
OLLAMA_LLAMA = "llama3.1"

# Convenience groupings
OPENAI_MODELS = [OPENAI_GPT4O, OPENAI_GPT4O_MINI]
GROQ_MODELS = [GROQ_LLAMA_70B, GROQ_LLAMA_8B]
MISTRAL_MODELS = [MISTRAL_LARGE, MISTRAL_SMALL, MISTRAL_CODESTRAL]
OLLAMA_MODELS = [OLLAMA_DOLPHIN, OLLAMA_LLAMA]
ALL_MODELS = OPENAI_MODELS + GROQ_MODELS + MISTRAL_MODELS + OLLAMA_MODELS

# ---------------------------------------------------------------------------
# Cost per 1 000 tokens  (USD)
# ---------------------------------------------------------------------------
COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    # OpenAI
    OPENAI_GPT4O:       {"input": 0.0025,  "output": 0.01},
    OPENAI_GPT4O_MINI:  {"input": 0.00015, "output": 0.0006},
    # Groq
    GROQ_LLAMA_70B:     {"input": 0.00059, "output": 0.00079},
    GROQ_LLAMA_8B:      {"input": 0.00005, "output": 0.00008},
    # Mistral
    MISTRAL_LARGE:      {"input": 0.002,   "output": 0.006},
    MISTRAL_SMALL:      {"input": 0.0002,  "output": 0.0006},
    MISTRAL_CODESTRAL:  {"input": 0.0003,  "output": 0.0009},
    # Ollama (free / local)
    OLLAMA_DOLPHIN:     {"input": 0.0,     "output": 0.0},
    OLLAMA_LLAMA:       {"input": 0.0,     "output": 0.0},
}

# ---------------------------------------------------------------------------
# Budget limits
# ---------------------------------------------------------------------------
DAILY_BUDGET_LIMIT: float = float(os.getenv("DAILY_BUDGET_LIMIT", "2.00"))
MONTHLY_BUDGET_LIMIT: float = float(os.getenv("MONTHLY_BUDGET_LIMIT", "30.00"))

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
