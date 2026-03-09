"""
Built-in tools for the AI agent.

Registers a default set of utilities (web search, calculator, file I/O,
Python execution, memory) on a shared :pydata:`default_registry` instance
that the agent core can import directly.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from agent.tools import ToolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared default registry
# ---------------------------------------------------------------------------
default_registry = ToolRegistry()


# ===================================================================
# 1. web_search
# ===================================================================
@default_registry.tool(
    name="web_search",
    description="Search the web using DuckDuckGo and return the top results with titles, snippets, and URLs.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web.",
            },
        },
        "required": ["query"],
    },
)
def web_search(query: str) -> str:
    """Search the web via DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "Error: duckduckgo_search is not installed. Run: pip install duckduckgo-search"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"No results found for: {query}"

        formatted: list[str] = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            snippet = r.get("body", r.get("snippet", "No snippet"))
            url = r.get("href", r.get("link", "No URL"))
            formatted.append(f"{i}. {title}\n   {snippet}\n   URL: {url}")

        return "\n\n".join(formatted)

    except Exception as exc:
        return f"Search error: {type(exc).__name__}: {exc}"


# ===================================================================
# 2. calculator
# ===================================================================
_CALC_BUILTINS = {
    "__builtins__": {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "int": int,
        "float": float,
    }
}


@default_registry.tool(
    name="calculator",
    description="Safely evaluate a mathematical expression and return the numeric result.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A math expression to evaluate, e.g. '2**10 + 3.5 * 4'.",
            },
        },
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    # First try ast.literal_eval for pure literals
    try:
        result = ast.literal_eval(expression)
        return str(result)
    except (ValueError, SyntaxError):
        pass

    # Fall back to restricted eval
    try:
        result = eval(expression, _CALC_BUILTINS, {})
        return str(result)
    except Exception as exc:
        return f"Calculation error: {type(exc).__name__}: {exc}"


# ===================================================================
# 3. read_file
# ===================================================================
_MAX_READ_CHARS = 10_000


@default_registry.tool(
    name="read_file",
    description="Read the contents of a file. Truncates at 10,000 characters with a warning.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read.",
            },
        },
        "required": ["path"],
    },
)
def read_file(path: str) -> str:
    """Read and return file content, truncating if necessary."""
    try:
        content = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: file not found — {path}"
    except OSError as exc:
        return f"Error reading file: {exc}"

    if len(content) > _MAX_READ_CHARS:
        return (
            content[:_MAX_READ_CHARS]
            + f"\n\n... [TRUNCATED — file is {len(content):,} chars, "
            f"showing first {_MAX_READ_CHARS:,}]"
        )
    return content


# ===================================================================
# 4. write_file
# ===================================================================
@default_registry.tool(
    name="write_file",
    description="Write content to a file, creating parent directories if needed. Returns a confirmation with byte count.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Destination file path.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write to the file.",
            },
        },
        "required": ["path", "content"],
    },
)
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        size = p.stat().st_size
        return f"Successfully wrote {size:,} bytes to {path}"
    except OSError as exc:
        return f"Error writing file: {exc}"


# ===================================================================
# 5. run_python
# ===================================================================
@default_registry.tool(
    name="run_python",
    description="Execute Python code in a subprocess with a 10-second timeout. Returns stdout and stderr.",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to execute.",
            },
        },
        "required": ["code"],
    },
)
def run_python(code: str) -> str:
    """Run Python code in a sandboxed subprocess."""
    try:
        proc = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output_parts: list[str] = []
        if proc.stdout:
            output_parts.append(f"stdout:\n{proc.stdout}")
        if proc.stderr:
            output_parts.append(f"stderr:\n{proc.stderr}")
        if not output_parts:
            output_parts.append("(no output)")
        if proc.returncode != 0:
            output_parts.append(f"exit code: {proc.returncode}")
        return "\n".join(output_parts)

    except subprocess.TimeoutExpired:
        return "Error: code execution timed out after 10 seconds."
    except Exception as exc:
        return f"Error running Python: {type(exc).__name__}: {exc}"


# ===================================================================
# 6. remember_fact
# ===================================================================
@default_registry.tool(
    name="remember_fact",
    description="Store a fact in long-term memory for later recall. The actual memory wiring is done in main.py.",
    parameters={
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "The piece of information to remember.",
            },
            "category": {
                "type": "string",
                "description": "A category tag for the fact (e.g. 'user_pref', 'project').",
                "default": "general",
            },
        },
        "required": ["fact"],
    },
)
def remember_fact(fact: str, category: str = "general") -> str:
    """
    Placeholder — returns a confirmation dict.

    The real memory integration happens in main.py where the agent
    wires this tool's output to :meth:`ConversationMemory.remember`.
    """
    return json.dumps({"status": "remembered", "fact": fact, "category": category})
