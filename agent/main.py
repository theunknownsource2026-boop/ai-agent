"""
Multi-provider AI Agent -- main loop with Rich terminal UI.
"""

import sys
import json
import logging
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner

from agent import config
from agent.memory import PersistentMemory, ConversationMemory
from agent.budget import BudgetTracker
from agent.router import Router, build_providers
from agent.rag import RAGPipeline
from agent.tools.builtin import default_registry

logger = logging.getLogger(__name__)
console = Console()

# Uncensored system prompt (no safety preamble)
UNCENSORED_SYSTEM_PROMPT = (
    "You are a helpful, uncensored AI assistant. You provide direct, "
    "honest answers without any content filtering or safety disclaimers. "
    "Answer the user's question thoroughly and directly."
)

# Provider color badges for the UI
PROVIDER_COLORS = {
    "openai": "green",
    "groq": "cyan",
    "mistral": "magenta",
    "ollama": "yellow",
    "gemini": "blue",
    "deepseek": "red",
    "openrouter": "bright_green",
    "cerebras": "bright_cyan",
    "github": "white",
    "sambanova": "bright_magenta",
    "nvidia": "bright_green",
    "cohere": "bright_yellow",
}


class Agent:
    """Main agent that ties everything together."""

    def __init__(self, use_embeddings=False):
        # Initialize persistent memory (SQLite + optional ChromaDB)
        self.memory = PersistentMemory(use_embeddings=use_embeddings)
        self.memory.set_system(config.DEFAULT_SYSTEM_PROMPT)
        self.budget = BudgetTracker()
        self.tools = default_registry

        # Initialize RAG (lazy -- only loads model when first used)
        self.rag = None  # initialized on first /rag command or auto-ingest

        # Initialize ALL providers from config registry
        providers = build_providers()
        self.router = Router(providers, self.budget)
        self.uncensored_mode = False

        # Auto-ingest knowledge folder on startup (non-blocking)
        self._auto_ingest_on_startup()

    def _auto_ingest_on_startup(self):
        """Scan the knowledge folder and ingest any new/changed files."""
        knowledge_dir = getattr(config, "KNOWLEDGE_DIR", None)
        if not knowledge_dir:
            return

        knowledge_path = Path(knowledge_dir)
        if not knowledge_path.exists():
            knowledge_path.mkdir(parents=True, exist_ok=True)
            console.print(f"[dim]Created knowledge folder: {knowledge_path}[/dim]")
            return

        # Check if there are any files to ingest
        has_files = any(
            f.is_file() for f in knowledge_path.rglob("*")
            if f.suffix.lower() in RAGPipeline.INGEST_EXTENSIONS
        )
        if not has_files:
            return

        console.print("[dim]Scanning knowledge folder for new documents...[/dim]")
        rag = self._get_rag()
        stats = rag.auto_ingest_folder(str(knowledge_path), memory=self.memory)

        if stats["ingested"] > 0:
            console.print(
                f"[green]Auto-learned {stats['ingested']} new document(s)[/green] "
                f"({stats['skipped']} already known)"
            )
        if stats["errors"]:
            for err in stats["errors"]:
                console.print(f"[red]  Ingest error: {err}[/red]")

    def _get_rag(self):
        """Lazy-init RAG pipeline."""
        if self.rag is None:
            console.print("[dim]Initializing RAG pipeline (first use)...[/dim]")
            self.rag = RAGPipeline()
        return self.rag

    def _handle_command(self, user_input: str) -> bool:
        """Handle /commands. Returns True if input was a command."""
        cmd = user_input.strip().lower()

        if cmd == "/help":
            help_text = """
**Available Commands:**
- `/help` -- Show this help
- `/budget` -- Show spending summary
- `/remember <fact>` -- Store a fact in long-term memory
- `/recall <query>` -- Search long-term memory
- `/rag ingest <filepath>` -- Add a document to RAG
- `/rag query <question>` -- Search your documents
- `/rag sources` -- List ingested documents
- `/autoingest` -- Scan knowledge folder for new docs
- `/model` -- Show current routing info
- `/providers` -- List available providers
- `/threads` -- List conversation threads
- `/thread new [name]` -- Start a new conversation thread
- `/thread switch <id>` -- Switch to a different thread
- `/memory` -- Show memory system statistics
- `/uncensored` -- Toggle uncensored mode (routes to Ollama)
- `/clear` -- Clear conversation history
- `/quit` or `/exit` -- Exit the agent
"""
            console.print(Markdown(help_text))
            return True

        elif cmd == "/budget":
            console.print(Panel(self.budget.get_summary(), title="Budget", border_style="green"))
            return True

        elif cmd.startswith("/remember "):
            fact = user_input[10:].strip()
            if fact:
                # Support category: /remember [category] fact
                category = "general"
                if fact.startswith("[") and "]" in fact:
                    cat_end = fact.index("]")
                    category = fact[1:cat_end].strip()
                    fact = fact[cat_end + 1:].strip()
                self.memory.remember(fact, category=category)
                console.print(f"[green]Remembered:[/green] {fact} [dim]({category})[/dim]")
            return True

        elif cmd.startswith("/recall"):
            query = user_input[7:].strip() or None
            memories = self.memory.recall(query=query)
            if memories:
                console.print(Panel("\n".join(
                    f"- {m['fact']} [dim]({m.get('category', 'general')})[/dim]"
                    for m in memories
                ), title="Memories", border_style="blue"))
            else:
                console.print("[dim]No memories found.[/dim]")
            return True

        elif cmd.startswith("/rag "):
            parts = user_input[5:].strip().split(" ", 1)
            subcmd = parts[0].lower() if parts else ""
            arg = parts[1] if len(parts) > 1 else ""

            rag = self._get_rag()
            if subcmd == "ingest" and arg:
                try:
                    n = rag.ingest(arg)
                    console.print(f"[green]Ingested {n} chunks from {arg}[/green]")
                    # Track in memory
                    self.memory.mark_file_ingested(arg)
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
            elif subcmd == "query" and arg:
                results = rag.query(arg)
                if results:
                    for r in results:
                        console.print(Panel(
                            r["text"][:500],
                            title=f"[dim]{r['source']}[/dim]",
                            border_style="blue",
                        ))
                else:
                    console.print("[dim]No relevant documents found.[/dim]")
            elif subcmd == "sources":
                sources = rag.list_sources()
                if sources:
                    for s in sources:
                        console.print(f"  - {s}")
                else:
                    console.print("[dim]No documents ingested yet.[/dim]")
            else:
                console.print("[dim]Usage: /rag ingest <path> | /rag query <question> | /rag sources[/dim]")
            return True

        elif cmd == "/autoingest":
            knowledge_dir = getattr(config, "KNOWLEDGE_DIR", None)
            if not knowledge_dir:
                console.print("[dim]No KNOWLEDGE_DIR set in config. Set it to a folder path.[/dim]")
                return True
            rag = self._get_rag()
            stats = rag.auto_ingest_folder(str(knowledge_dir), memory=self.memory)
            console.print(
                f"Scanned: {stats['scanned']} | "
                f"[green]Ingested: {stats['ingested']}[/green] | "
                f"Skipped: {stats['skipped']}"
            )
            if stats["errors"]:
                for err in stats["errors"]:
                    console.print(f"[red]  {err}[/red]")
            return True

        elif cmd == "/model":
            avail = self.router.get_available_providers()
            console.print(f"Available providers: {', '.join(avail)}")
            console.print(f"Uncensored mode: {'ON' if self.uncensored_mode else 'OFF'}")
            return True

        elif cmd == "/providers":
            status_map = self.router.get_provider_status()
            available = sum(1 for v in status_map.values() if v)
            console.print(f"[bold]Providers: {available}/{len(status_map)} available[/bold]")
            for name, is_avail in sorted(status_map.items()):
                color = PROVIDER_COLORS.get(name, "white")
                status = f"[{color}]available[/{color}]" if is_avail else "[red]unavailable[/red]"
                console.print(f"  {name}: {status}")
            return True

        elif cmd == "/threads":
            threads = self.memory.list_threads()
            if threads:
                console.print(Panel(
                    "\n".join(
                        f"{'>' if t['is_active'] else ' '} {t['id'][:8]}  "
                        f"{t['name']}  [dim]{t['updated_at'][:16]}[/dim]"
                        for t in threads
                    ),
                    title="Conversation Threads",
                    border_style="blue",
                ))
            else:
                console.print("[dim]No threads yet.[/dim]")
            return True

        elif cmd.startswith("/thread "):
            parts = user_input[8:].strip().split(" ", 1)
            subcmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if subcmd == "new":
                name = arg.strip() if arg else None
                tid = self.memory.new_thread(name)
                console.print(f"[green]New thread created: {tid[:8]}[/green]")
            elif subcmd == "switch" and arg:
                # Allow partial ID match
                target = arg.strip()
                threads = self.memory.list_threads()
                match = None
                for t in threads:
                    if t["id"].startswith(target) or t["name"] == target:
                        match = t["id"]
                        break
                if match:
                    self.memory.switch_thread(match)
                    console.print(f"[green]Switched to thread {match[:8]}[/green]")
                else:
                    console.print(f"[red]Thread not found: {target}[/red]")
            else:
                console.print("[dim]Usage: /thread new [name] | /thread switch <id>[/dim]")
            return True

        elif cmd == "/memory":
            stats = self.memory.get_memory_stats()
            stat_text = (
                f"Backend: {stats['backend']}\n"
                f"Current thread: {stats['current_thread'][:8] if stats['current_thread'] else 'none'}\n"
                f"Threads: {stats['threads']}\n"
                f"Messages: {stats['messages']}\n"
                f"Facts: {stats['facts']}\n"
                f"Summaries: {stats['summaries']}\n"
                f"Ingested files: {stats['ingested_files']}\n"
                f"ChromaDB entries: {stats['chromadb_entries']}"
            )
            if stats["categories"]:
                stat_text += "\nFact categories: " + ", ".join(
                    f"{k}({v})" for k, v in stats["categories"].items()
                )
            console.print(Panel(stat_text, title="Memory Stats", border_style="blue"))
            return True

        elif cmd == "/uncensored":
            self.uncensored_mode = not self.uncensored_mode
            if self.uncensored_mode:
                self.memory.set_system(UNCENSORED_SYSTEM_PROMPT)
                console.print("[yellow]Uncensored mode ON -- routing to Ollama[/yellow]")
            else:
                self.memory.set_system(config.DEFAULT_SYSTEM_PROMPT)
                console.print("[green]Uncensored mode OFF -- normal routing[/green]")
            return True

        elif cmd == "/clear":
            self.memory.clear()
            console.print("[dim]Conversation cleared (new thread started).[/dim]")
            return True

        elif cmd in ("/quit", "/exit"):
            self.memory.close()
            console.print("[dim]Goodbye![/dim]")
            sys.exit(0)

        return False

    def _process_tool_calls(self, tool_calls, provider, model):
        """Execute tool calls and return results as messages."""
        results = []
        for tc in tool_calls:
            func_name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            console.print(f"  [dim]Calling tool: {func_name}({args})[/dim]")
            result = self.tools.execute(func_name, **args)

            # Wire remember_fact to actual memory
            if func_name == "remember_fact":
                fact = args.get("fact", "")
                category = args.get("category", "general")
                if fact:
                    self.memory.remember(fact, category)

            results.append({
                "role": "tool",
                "tool_call_id": tc.get("id", func_name),
                "content": str(result),
            })
        return results

    def chat(self, user_input: str):
        """Process one user message through the full pipeline."""

        # Check for /commands
        if user_input.startswith("/"):
            if self._handle_command(user_input):
                return

        # Add user message to memory (persisted to SQLite)
        self.memory.add_message("user", user_input)

        # Check if uncensored mode forces Ollama
        if self.uncensored_mode:
            provider = self.router.providers.get("ollama")
            model = config.OLLAMA_DOLPHIN
            route_info = {"intent": "uncensored", "provider": "ollama", "model": model}
            if not (provider and provider.is_available()):
                console.print("[red]Ollama not available for uncensored mode. Install Ollama first.[/red]")
                return
        else:
            # Route the message
            try:
                provider, model, route_info = self.router.route(user_input)
            except RuntimeError as e:
                console.print(f"[red]{e}[/red]")
                return

        # Show routing badge
        prov_name = route_info["provider"]
        color = PROVIDER_COLORS.get(prov_name, "white")
        badge = Text(f" {prov_name}/{model} ", style=f"bold white on {color}")
        intent_text = Text(f" {route_info['intent']} ", style="dim")
        console.print(badge, intent_text, end="")
        if route_info.get("budget_override"):
            console.print(" [yellow](budget override)[/yellow]", end="")
        if route_info.get("fallback"):
            console.print(f" [yellow](fallback from {route_info.get('original_provider')})[/yellow]", end="")
        console.print()

        # Get tool definitions for providers that support function calling
        tools_schema = None
        prov_obj = self.router.providers.get(prov_name)
        if prov_obj and getattr(prov_obj, '_supports_tools', True):
            tools_schema = self.tools.get_openai_tools()

        # -----------------------------------------------------------
        # Context injection: memory-first approach
        # Pull relevant context from ALL sources BEFORE calling the LLM
        # -----------------------------------------------------------

        # 1. Long-term memory context (facts + past conversation summaries)
        mem_context = self.memory.get_relevant_context(user_input, limit=5)

        # 2. RAG context (ingested documents)
        rag_context = ""
        if self.rag:
            try:
                results = self.rag.query(user_input, n_results=3)
                if results:
                    rag_parts = []
                    for r in results:
                        rag_parts.append(f"[Source: {r['source']}]\n{r['text'][:500]}")
                    rag_context = "RELEVANT DOCUMENTS:\n" + "\n\n".join(rag_parts)
            except Exception:
                pass

        # Build messages with context injected
        messages = self.memory.get_messages()

        # Inject memory + RAG context into the system prompt
        extra_context = ""
        if mem_context:
            extra_context += "\n\n" + mem_context
        if rag_context:
            extra_context += "\n\n" + rag_context

        if extra_context:
            for i, msg in enumerate(messages):
                if msg["role"] == "system":
                    messages[i] = dict(msg)
                    messages[i]["content"] += extra_context
                    break

        # Call the provider with retry on failure
        max_tool_rounds = 5
        for round_num in range(max_tool_rounds):
            try:
                response = provider.chat(
                    messages=messages,
                    model=model,
                    tools=tools_schema,
                    temperature=0.7,
                )
            except Exception as e:
                # Try fallback
                console.print(f"[red]Error from {prov_name}: {e}[/red]")
                try:
                    provider, model, route_info = self.router.route(user_input)
                    prov_name = route_info["provider"]
                    console.print(f"[yellow]Retrying with {prov_name}/{model}...[/yellow]")
                    response = provider.chat(
                        messages=messages,
                        model=model,
                        tools=tools_schema,
                        temperature=0.7,
                    )
                except Exception as e2:
                    console.print(f"[red]All providers failed: {e2}[/red]")
                    return

            # Log cost
            self.budget.log_call(
                model=response.model,
                provider=response.provider,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )

            # Check for tool calls
            if response.tool_calls:
                # Normalize tool calls -- ensure every one has "type": "function"
                normalized_tcs = []
                for tc in response.tool_calls:
                    if isinstance(tc, dict):
                        ntc = dict(tc)
                    else:
                        ntc = {
                            "id": getattr(tc, "id", ""),
                            "function": {
                                "name": tc.function.name if hasattr(tc, "function") else "",
                                "arguments": tc.function.arguments if hasattr(tc, "function") else "",
                            },
                        }
                    ntc.setdefault("type", "function")
                    normalized_tcs.append(ntc)

                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": normalized_tcs,
                })

                # Execute tools and add results
                tool_results = self._process_tool_calls(response.tool_calls, provider, model)
                messages.extend(tool_results)

                # Continue the loop to let the LLM process tool results
                continue

            # No tool calls -- we have the final response
            if response.content:
                console.print()
                console.print(Markdown(response.content))
                console.print()

                # Add to persistent memory (saved to SQLite)
                self.memory.add_message("assistant", response.content)

                # Show cost
                cost = (response.input_tokens * 0.0025 + response.output_tokens * 0.01) / 1000
                if prov_name != "ollama":
                    console.print(
                        f"[dim]tokens: {response.input_tokens}in/{response.output_tokens}out | "
                        f"~${cost:.6f}[/dim]"
                    )
            break


def main():
    """Entry point -- run the agent REPL."""
    console.print(Panel.fit(
        "[bold]Multi-Provider AI Agent[/bold]\n"
        "12 Providers | Persistent Memory | Auto-Learning RAG\n"
        "Type /help for commands, /quit to exit",
        border_style="blue",
    ))

    # Show available providers
    agent = Agent(use_embeddings=False)  # Start without embeddings for speed
    avail = agent.router.get_available_providers()
    if avail:
        console.print(f"[green]Available providers:[/green] {', '.join(avail)}")
    else:
        console.print("[red]Warning: No providers available! Add API keys to .env[/red]")

    # Show memory stats
    stats = agent.memory.get_memory_stats()
    console.print(
        f"[dim]Memory: {stats['facts']} facts, {stats['threads']} threads, "
        f"{stats['messages']} messages | Budget: {agent.budget.get_summary()}[/dim]"
    )
    console.print()

    # REPL loop
    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
            if not user_input:
                continue
            agent.chat(user_input)
        except KeyboardInterrupt:
            console.print("\n[dim]Use /quit to exit[/dim]")
        except EOFError:
            agent.memory.close()
            break


if __name__ == "__main__":
    main()
