#!/usr/bin/env python3
"""
Launch the AI Agent.

Usage:
    python run.py          # Start the interactive agent
    python run.py --test   # Test which providers are available
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_providers():
    """Quick connectivity test for all providers."""
    from rich.console import Console
    from rich.table import Table
    console = Console()

    from agent.providers.openai_provider import OpenAIProvider
    from agent.providers.groq_provider import GroqProvider
    from agent.providers.mistral_provider import MistralProvider
    from agent.providers.ollama_provider import OllamaProvider

    table = Table(title="Provider Status")
    table.add_column("Provider", style="bold")
    table.add_column("Status")
    table.add_column("Default Model")

    for name, ProvClass in [
        ("OpenAI", OpenAIProvider),
        ("Groq", GroqProvider),
        ("Mistral", MistralProvider),
        ("Ollama", OllamaProvider),
    ]:
        try:
            p = ProvClass()
            if p.is_available():
                table.add_row(name, "[green]Available[/green]", p._default_model)
            else:
                table.add_row(name, "[yellow]Not configured[/yellow]", "-")
        except Exception as e:
            table.add_row(name, f"[red]Error: {e}[/red]", "-")

    console.print(table)

    from agent.budget import BudgetTracker
    bt = BudgetTracker()
    console.print(f"\nBudget: {bt.get_summary()}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_providers()
    else:
        from agent.main import main
        main()
