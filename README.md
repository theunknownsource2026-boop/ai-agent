# Multi-Provider AI Agent

A modular, multi-provider AI agent with intelligent routing, tool calling,
RAG (Retrieval-Augmented Generation), persistent memory, budget tracking,
and a Rich terminal UI.

## Features

- **Multi-Provider Routing** -- Automatically routes queries to the best
  provider (OpenAI, Groq, Mistral, or Ollama) based on intent classification
- **Smart Intent Detection** -- Keyword-based classification routes code
  questions to Codestral, creative tasks to GPT-4o, general chat to Groq
  (fast), and uncensored queries to Ollama (local)
- **Tool Calling** -- Built-in tools for web search (DuckDuckGo), calculator,
  file I/O, Python execution, and memory storage
- **RAG Pipeline** -- Ingest documents (PDF, Markdown, code), chunk and embed
  with sentence-transformers, query with ChromaDB vector search
- **Conversation Memory** -- Sliding window (20 messages) for context, plus
  persistent long-term fact storage with semantic or keyword recall
- **Budget Tracking** -- Per-call cost logging to JSONL, daily/monthly
  spending limits, automatic fallback to free Ollama when over budget
- **Uncensored Mode** -- Toggle to route all queries through a local Ollama
  model (e.g., dolphin-mistral) with no content filtering
- **Rich Terminal UI** -- Color-coded provider badges, Markdown rendering,
  tool call indicators, and cost display per response
- **Graceful Fallbacks** -- If a provider is unavailable or errors out, the
  router walks a fallback chain (Groq -> Mistral -> OpenAI -> Ollama)

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
python run.py
```

## Getting API Keys

| Provider | Console URL | Free Tier |
|----------|------------|-----------|
| **OpenAI** | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | $5 credit for new accounts |
| **Groq** | [console.groq.com/keys](https://console.groq.com/keys) | Generous free tier |
| **Mistral** | [console.mistral.ai/api-keys](https://console.mistral.ai/api-keys) | Free tier available |
| **Ollama** | [ollama.com/download](https://ollama.com/download) | Fully free (local) |

### Ollama Setup (Local Models)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull recommended models
ollama pull llama3.1
ollama pull dolphin-mistral    # For uncensored mode
```

## Commands Reference

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/budget` | Display spending summary (daily/monthly) |
| `/remember <fact>` | Store a fact in long-term memory |
| `/recall [query]` | Search long-term memory (semantic or keyword) |
| `/rag ingest <path>` | Ingest a document into the RAG pipeline |
| `/rag query <question>` | Search ingested documents |
| `/rag sources` | List all ingested document sources |
| `/model` | Show current routing info and available providers |
| `/providers` | List all providers with availability status |
| `/uncensored` | Toggle uncensored mode (routes to local Ollama) |
| `/clear` | Clear conversation history (keeps long-term memory) |
| `/quit` or `/exit` | Exit the agent |

## Architecture

```
agent_project/
|-- run.py                          # Entry point (python run.py)
|-- requirements.txt                # pip dependencies
|-- .env.example                    # Template for API keys
|-- README.md                       # This file
|-- costs.jsonl                     # Auto-generated spending log
|-- memory.jsonl                    # Auto-generated long-term memory
|
|-- agent/
    |-- __init__.py                 # Package marker (v1.0.0)
    |-- config.py                   # Env vars, model constants, pricing, budget limits
    |-- main.py                     # Agent class + Rich REPL (the big one)
    |-- router.py                   # Intent classifier + provider routing + fallbacks
    |-- budget.py                   # JSONL cost tracker, daily/monthly limits
    |-- memory.py                   # Sliding window + ChromaDB long-term memory
    |-- rag.py                      # Document chunking, embedding, vector retrieval
    |
    |-- providers/
    |   |-- __init__.py             # ChatResponse dataclass + BaseProvider ABC
    |   |-- openai_provider.py      # OpenAI SDK: chat, streaming, tool_calls
    |   |-- groq_provider.py        # Groq SDK: same interface, x_groq usage
    |   |-- mistral_provider.py     # Mistral SDK: client.chat.complete/stream
    |   |-- ollama_provider.py      # Raw httpx to Ollama REST, NDJSON streaming
    |
    |-- tools/
        |-- __init__.py             # ToolRegistry class (register, export, execute)
        |-- builtin.py              # Default tools: web_search, calculator, file I/O, etc.
```

### Module Responsibilities

| Module | What It Does |
|--------|-------------|
| `config.py` | Loads `.env`, defines model name constants (9 models across 4 providers), pricing dict, budget limits, default system prompt |
| `main.py` | Ties everything together: Agent class with REPL loop, `/command` handling, provider routing, tool call execution, RAG/memory context injection, Rich UI rendering |
| `router.py` | Classifies user intent (code/creative/reasoning/uncensored/chat) via keyword matching, maps to optimal provider/model, handles budget overflow and provider fallbacks |
| `budget.py` | Logs every API call to `costs.jsonl` with timestamps, calculates daily/monthly spend, enforces configurable limits |
| `memory.py` | Manages sliding window of recent messages (default 20) plus persistent JSONL fact store; optionally uses ChromaDB + sentence-transformers for semantic recall |
| `rag.py` | Loads files (PDF, Markdown, code), chunks text with overlap, embeds via sentence-transformers, stores/queries in ChromaDB |
| `providers/*.py` | Each provider wraps its SDK behind the uniform `BaseProvider` interface (`chat()`, `stream_chat()`, `is_available()`) returning `ChatResponse` dataclasses |
| `tools/*.py` | `ToolRegistry` handles registration, OpenAI-format export, and safe execution; `builtin.py` registers 6 default tools |

## Cost Estimates

Approximate costs per 1,000 tokens (USD):

| Model | Input | Output | Best For |
|-------|-------|--------|----------|
| `gpt-4o` | $0.0025 | $0.010 | Creative writing, complex reasoning |
| `gpt-4o-mini` | $0.00015 | $0.0006 | General fallback, cost-effective |
| `llama-3.1-70b` (Groq) | $0.00059 | $0.00079 | Fast reasoning, code review |
| `llama-3.1-8b` (Groq) | $0.00005 | $0.00008 | Ultra-fast chat, lowest cost |
| `mistral-large` | $0.002 | $0.006 | Complex tasks, multilingual |
| `mistral-small` | $0.0002 | $0.0006 | Fast general purpose |
| `codestral` | $0.0003 | $0.0009 | Code generation and review |
| `dolphin-mistral` (Ollama) | Free | Free | Uncensored, local, private |
| `llama3.1` (Ollama) | Free | Free | Local fallback, no API needed |

With default budget limits ($2/day, $30/month), typical usage costs:
- **Light usage** (~50 queries/day): ~$0.10-0.50/day
- **Heavy usage** (~200 queries/day): ~$1.00-2.00/day
- **Over budget**: Automatically falls back to free Ollama models

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
