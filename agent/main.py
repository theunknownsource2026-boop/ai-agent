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
from agent.memory import ConversationMemory
from agent.budget import BudgetTracker
from agent.router import Router
from agent.rag import RAGPipeline
from agent.tools.builtin import default_registry
from agent.providers.openai_provider import OpenAIProvider
from agent.providers.groq_provider import GroqProvider
from agent.providers.mistral_provider import MistralProvider
from agent.providers.ollama_provider import OllamaProvider

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
}


class Agent:
    """Main agent that ties everything together."""

    def __init__(self, use_embeddings=False):
        # Initialize components
        self.memory = ConversationMemory(use_embeddings=use_embeddings)
        self.memory.set_system(config.DEFAULT_SYSTEM_PROMPT)
        self.budget = BudgetTracker()
        self.tools = default_registry

        # Initialize RAG (lazy -- only loads model when first used)
        self.rag = None  # initialized on first /rag command

        # Initialize providers
        providers = {}
        for name, ProvClass in [
            ("openai", OpenAIProvider),
            ("groq", GroqProvider),
            ("mistral", MistralProvider),
            ("ollama", OllamaProvider),
        ]:
            try:
                providers[name] = ProvClass()
            except Exception as e:
                logger.warning("Could not init %s: %s", name, e)

        self.router = Router(providers, self.budget)
        self.uncensored_mode = False

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
- `/model` -- Show current routing info
- `/providers` -- List available providers
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
                self.memory.remember(fact)
                console.print(f"[green]Remembered:[/green] {fact}")
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

        elif cmd == "/model":
            avail = self.router.get_available_providers()
            console.print(f"Available providers: {', '.join(avail)}")
            console.print(f"Uncensored mode: {'ON' if self.uncensored_mode else 'OFF'}")
            return True

        elif cmd == "/providers":
            for name in ["openai", "groq", "mistral", "ollama"]:
                prov = self.router.providers.get(name)
                status = "[green]available[/green]" if (prov and prov.is_available()) else "[red]unavailable[/red]"
                console.print(f"  {name}: {status}")
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
            console.print("[dim]Conversation cleared.[/dim]")
            return True

        elif cmd in ("/quit", "/exit"):
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

        # Add user message to memory
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
        if prov_name in ("openai", "groq", "mistral"):
            tools_schema = self.tools.get_openai_tools()

        # RAG context injection: if RAG is initialized, search for relevant docs
        rag_context = ""
        if self.rag:
            try:
                results = self.rag.query(user_input, n_results=3)
                if results:
                    rag_context = "\n\n---\nRelevant context from your documents:\n"
                    for r in results:
                        rag_context += f"\n[Source: {r['source']}]\n{r['text'][:500]}\n"
            except Exception:
                pass

        # Memory context: recall relevant long-term memories
        mem_context = ""
        try:
            memories = self.memory.recall(query=user_input, limit=3)
            if memories:
                mem_context = "\n\n---\nRelevant memories:\n"
                for m in memories:
                    mem_context += f"- {m['fact']}\n"
        except Exception:
            pass

        # Build messages with context injected into system prompt
        messages = self.memory.get_messages()
        if rag_context or mem_context:
            # Inject context into system message
            for i, msg in enumerate(messages):
                if msg["role"] == "system":
                    messages[i] = dict(msg)
                    messages[i]["content"] += rag_context + mem_context
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
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": response.tool_calls,
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

                # Add to memory
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
        "OpenAI + Groq + Mistral + Ollama\n"
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

    console.print(f"[dim]Budget: {agent.budget.get_summary()}[/dim]")
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
            break


if __name__ == "__main__":
    main()
