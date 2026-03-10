# Multi-Provider AI Agent

A modular, multi-provider AI agent with intelligent routing, tool calling,
RAG (Retrieval-Augmented Generation), persistent memory, budget tracking,
and both CLI + Web UI interfaces.

**14 providers supported. 7 are 100% free. FREE_MODE on by default = $0 spend.**

## Features

- **14-Provider Smart Routing** -- Automatically routes queries to the best
  available provider based on intent classification (code, reasoning, creative,
  chat, uncensored)
- **FREE_MODE Safety Lock** -- On by default. Guarantees $0 spend by blocking
  all paid providers. Set `FREE_MODE=false` in `.env` when you're ready to
  use credit-based providers
- **Smart Intent Detection** -- Keyword-based classification routes code
  questions to Codestral/Groq, creative tasks to Gemini Pro, reasoning to
  QwQ/DeepSeek R1, general chat to Groq (fastest), and uncensored to Ollama
- **Tool Calling** -- Built-in tools for web search (DuckDuckGo), calculator,
  file I/O, Python execution, and memory storage
- **RAG Pipeline** -- Ingest documents (PDF, Markdown, code), chunk and embed
  with sentence-transformers, query with ChromaDB vector search
- **Conversation Memory** -- Thread-based with SQLite persistence, forget
  command, thread rename/delete, and memory stats
- **Budget Tracking** -- Per-call cost logging, daily/monthly spending limits,
  per-provider credit tracking, automatic fallback to free providers
- **Uncensored Mode** -- Route all queries through local Ollama (dolphin-llama3)
- **Web UI** -- Flask dashboard with thread sidebar, memory panel, RAG upload,
  provider status, and all CLI features in a browser
- **CLI with Rich UI** -- Color-coded provider badges, Markdown rendering,
  tool call indicators, and cost display per response

## Quick Start

### 1. Clone or download the project

```bash
git clone <your-repo-url> agent_project
cd agent_project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

### 4. Verify providers

```bash
python run.py --test
```

This will show a table of which providers are configured and available.

### 5. Start the agent

```bash
# CLI mode:
python run.py

# Web UI mode:
python web_ui.py
# Then open http://localhost:5000
```

---

## Free Tier Cheat Sheet

This is the real deal -- verified March 2026. These are the actual limits
you get for $0.

### 100% FREE Providers (no credit card, no spend, ever)

| Provider | Free Limit | Rate Limit | Best For |
|----------|-----------|------------|----------|
| **Groq** | Unlimited tokens | 30 RPM, ~1K req/day | Fastest inference, general chat |
| **Google Gemini** | Unlimited tokens | 15 RPM, 250K TPM | Reasoning, creative, code |
| **Cerebras** | Unlimited tokens | Free tier | Fastest inference on earth |
| **OpenRouter** | Free on `:free` models | 20 RPM, 50 req/day | Access to free DeepSeek R1, Llama 4 |
| **Ollama** | Unlimited (local) | Your hardware | Uncensored, private, offline |
| **Mistral** | ~1B tokens/month | 1 RPS, 500K TPM | Code (Codestral), European models |
| **Cohere** | 1,000 calls/month | 20 RPM | Trial key, non-commercial use |

### Credit-Based Providers (free until credits run out)

| Provider | Free Credits | After Credits | Cheapest Model |
|----------|-------------|---------------|----------------|
| **xAI (Grok)** | $25 signup + $150/mo (data sharing opt-in) | $0.10-$0.20/1M input | grok-3-mini |
| **Anthropic** | $5 signup credit | $1.00/$5.00 per 1M (Haiku) | claude-haiku-4.5 |
| **DeepSeek** | No free credits | $0.14/$0.28 per 1M | deepseek-chat (extremely cheap) |
| **OpenAI** | 3 RPM free tier (very limited) | $0.15/$0.60 per 1M (mini) | gpt-4o-mini |
| **Together AI** | NO free tier | $0.06/1M (Llama 3B) | Llama-3.2-3B-Instruct |

### How FREE_MODE Works

```
FREE_MODE=true  (default)
  Router ONLY picks from: Groq, Gemini, Cerebras, OpenRouter, Ollama, Mistral, Cohere
  Paid providers are registered but BLOCKED -- you literally cannot spend money
  
FREE_MODE=false
  All 14 providers unlocked
  Budget caps still apply: $0.50/day, $5.00/month (configurable)
  Over budget -> automatically falls back to free providers
```

### Routing Priority (FREE_MODE=true)

| Intent | Provider Chain (first available wins) |
|--------|--------------------------------------|
| **Code** | Mistral Codestral -> Groq Llama 70B -> Cerebras -> Gemini Flash -> OpenRouter DeepSeek |
| **Reasoning** | Groq QwQ-32B -> Gemini Pro -> Cerebras Qwen -> OpenRouter DeepSeek R1 -> Mistral Large |
| **Creative** | Gemini Pro -> Groq Llama 70B -> Cohere Command R+ -> Mistral Large -> Cerebras |
| **Chat** | Groq Llama 70B -> Cerebras -> Gemini Flash -> Mistral Small -> OpenRouter Auto -> Cohere |
| **Uncensored** | Ollama Dolphin (local only) |

---

## Getting API Keys

### Free providers (get these first)

| Provider | Signup URL | Notes |
|----------|-----------|-------|
| **Groq** | [console.groq.com/keys](https://console.groq.com/keys) | Instant, no credit card |
| **Google Gemini** | [aistudio.google.com](https://aistudio.google.com) | Click "Get API key" in sidebar |
| **Cerebras** | [cloud.cerebras.ai](https://cloud.cerebras.ai) | Free tier, fastest inference |
| **OpenRouter** | [openrouter.ai/keys](https://openrouter.ai/keys) | Stick to `:free` models |
| **Mistral** | [console.mistral.ai/api-keys](https://console.mistral.ai/api-keys) | Free tier, 1B tokens/mo |
| **Cohere** | [dashboard.cohere.com](https://dashboard.cohere.com/welcome/register) | Trial key, 1K calls/mo |
| **Ollama** | [ollama.com/download](https://ollama.com/download) | Local install, then `ollama serve` |

### Credit-based providers (optional, blocked by FREE_MODE)

| Provider | Signup URL | Free Credits |
|----------|-----------|-------------|
| **xAI (Grok)** | [console.x.ai](https://console.x.ai) | $25 signup credit |
| **Anthropic** | [console.anthropic.com](https://console.anthropic.com/settings/keys) | $5 signup credit |
| **DeepSeek** | [platform.deepseek.com](https://platform.deepseek.com/apikeys) | None (but dirt cheap) |
| **OpenAI** | [platform.openai.com](https://platform.openai.com/api-keys) | 3 RPM free (limited) |
| **Together AI** | [api.together.xyz](https://api.together.xyz/settings/api-keys) | None ($5 min purchase) |

### Ollama Setup (Local Models)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull recommended models
ollama pull llama3.1
ollama pull dolphin-llama3    # For uncensored mode
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/budget` | Display spending summary (daily/monthly) |
| `/remember <fact>` | Store a fact in long-term memory |
| `/recall [query]` | Search long-term memory |
| `/forget <text>` | Remove specific memory or conversation context |
| `/summary` | Summarize and compress current thread |
| `/threads` | List all conversation threads |
| `/thread new [name]` | Create a new thread |
| `/thread switch <name>` | Switch to an existing thread |
| `/thread rename <new>` | Rename current thread |
| `/thread delete <name>` | Delete a thread |
| `/memory` | Show memory stats |
| `/rag ingest <path>` | Ingest a document into the RAG pipeline |
| `/rag query <question>` | Search ingested documents |
| `/rag sources` | List all ingested document sources |
| `/model` | Show current routing info and available providers |
| `/providers` | List all providers with availability status |
| `/uncensored` | Toggle uncensored mode (routes to local Ollama) |
| `/clear` | Clear conversation history (keeps long-term memory) |
| `/quit` or `/exit` | Exit the agent |

---

## Architecture

```
agent_project/
|-- run.py                          # CLI entry point
|-- web_ui.py                       # Flask web UI (http://localhost:5000)
|-- requirements.txt                # pip dependencies
|-- .env.example                    # Template for all 11 API keys + FREE_MODE
|-- README.md                       # This file
|
|-- agent/
|   |-- __init__.py                 # Package marker
|   |-- config.py                   # 14 providers, FREE_MODE, costs, routing tables
|   |-- main.py                     # Agent class + Rich CLI REPL
|   |-- router.py                   # Intent classifier + FREE_MODE enforcer + fallbacks
|   |-- budget.py                   # Cost tracker, daily/monthly limits
|   |-- memory.py                   # SQLite threads, forget, rename, delete, stats
|   |-- rag.py                      # Document chunking, embedding, vector retrieval
|   |
|   |-- providers/
|   |   |-- __init__.py             # ChatResponse dataclass + BaseProvider ABC
|   |   |-- openai_provider.py      # OpenAI SDK
|   |   |-- groq_provider.py        # Groq SDK
|   |   |-- mistral_provider.py     # Mistral SDK
|   |   |-- ollama_provider.py      # Raw httpx to Ollama REST
|   |   |-- universal_provider.py   # Generic OpenAI-compatible (handles 10+ providers)
|   |
|   |-- tools/
|       |-- __init__.py             # ToolRegistry class
|       |-- builtin.py              # Default tools: web search, calc, file I/O, etc.
|
|-- templates/
    |-- dashboard.html              # Web UI: threads, memory, RAG, provider status
```

### Provider Architecture

All 14 providers use the same `BaseProvider` interface. Most are OpenAI-compatible
and handled by `UniversalProvider` -- only Groq, Mistral, and Ollama have dedicated
provider files for SDK-specific features.

```
User Message
    |
    v
Intent Classifier (code/reasoning/creative/chat/uncensored)
    |
    v
FREE_MODE check --> blocks paid providers if true
    |
    v
Budget check --> blocks costly models if over limit
    |
    v
Walk provider chain --> first available + eligible wins
    |
    v
Fallback chain --> 7 free providers, then 5 paid (if allowed)
    |
    v
Response + cost logged
```

---

## Cost Estimates (when FREE_MODE=false)

| Model | Provider | Input/1K | Output/1K | Notes |
|-------|----------|----------|-----------|-------|
| Groq Llama 70B | Groq | $0.00 | $0.00 | Free, fastest |
| Gemini Flash | Google | $0.00 | $0.00 | Free, 15 RPM |
| Cerebras Llama 70B | Cerebras | $0.00 | $0.00 | Free, ultra-fast |
| OpenRouter :free | OpenRouter | $0.00 | $0.00 | Free models only |
| Mistral Small | Mistral | $0.00 | $0.00 | Free tier |
| Cohere Command R | Cohere | $0.00 | $0.00 | Free trial |
| Ollama (any) | Local | $0.00 | $0.00 | Your hardware |
| DeepSeek Chat | DeepSeek | $0.00014 | $0.00028 | Near-free |
| Grok 3 Mini | xAI | $0.0001 | $0.0003 | Uses $25 credit |
| GPT-4o-mini | OpenAI | $0.00015 | $0.0006 | Pay-as-you-go |
| Claude Haiku 4.5 | Anthropic | $0.001 | $0.005 | Uses $5 credit |
| Together Llama 3B | Together | $0.00006 | $0.00006 | $5 min purchase |

With `FREE_MODE=true` (default), your cost is always **$0.00**.

With `FREE_MODE=false` and budget caps ($0.50/day, $5.00/month):
- **Light usage** (~100 queries/day): $0.00-0.05/day (mostly free providers)
- **Heavy usage** (~500 queries/day): $0.10-0.50/day (if paid providers needed)
- **Over budget**: Automatically falls back to free providers only

---

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
