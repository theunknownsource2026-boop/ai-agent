"""
Tool Registry — register, discover, and execute agent tools.

Provides a :class:`ToolRegistry` that stores callable tools with their
JSON-schema parameter definitions, and can export them in OpenAI
function-calling format.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PY_TYPE_TO_JSON: Dict[type, str] = {
    str:   "string",
    int:   "integer",
    float: "number",
    bool:  "boolean",
    list:  "array",
    dict:  "object",
}


def python_type_to_json_schema(type_hint: Any) -> str:
    """
    Map a Python type annotation to its JSON-schema type string.

    Supports bare types (``str``, ``int``, …) and falls back to
    ``"string"`` for anything unrecognized.
    """
    # Unwrap Optional[X] -> X (typing.Union[X, None])
    origin = getattr(type_hint, "__origin__", None)
    if origin is not None:
        args = getattr(type_hint, "__args__", ())
        # Optional[X] is Union[X, None]
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            type_hint = non_none[0]

    return _PY_TYPE_TO_JSON.get(type_hint, "string")


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Register, discover, and execute agent tools."""

    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, Any],
    ) -> None:
        """
        Register a tool manually.

        Parameters
        ----------
        name : str
            Unique tool name (used in function-calling payloads).
        function : callable
            The Python function to invoke.
        description : str
            One-line description shown to the model.
        parameters : dict
            JSON-schema object describing the expected arguments.
        """
        self.tools[name] = {
            "function": function,
            "description": description,
            "parameters": parameters,
        }
        logger.debug("Registered tool: %s", name)

    def tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
    ) -> Callable:
        """
        Decorator that registers a function as a tool.

        Usage::

            @registry.tool(
                name="web_search",
                description="Search the web.",
                parameters={"type": "object", "properties": {...}, "required": [...]},
            )
            def web_search(query: str) -> str:
                ...
        """
        def decorator(fn: Callable) -> Callable:
            self.register(name, fn, description, parameters)

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return fn(*args, **kwargs)
            return wrapper
        return decorator

    # ------------------------------------------------------------------
    # Lookup & export
    # ------------------------------------------------------------------

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Return the tool entry dict, or *None* if not found."""
        return self.tools.get(name)

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Export registered tools in OpenAI function-calling format.

        Returns
        -------
        list[dict]
            ``[{"type": "function", "function": {"name": …, "description": …, "parameters": …}}, …]``
        """
        result: List[Dict[str, Any]] = []
        for tool_name, entry in self.tools.items():
            result.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": entry["description"],
                    "parameters": entry["parameters"],
                },
            })
        return result

    def list_tools(self) -> List[str]:
        """Return the names of all registered tools."""
        return list(self.tools.keys())

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, name: str, **kwargs: Any) -> str:
        """
        Call a registered tool and return its result as a string.

        Any exception raised by the underlying function is caught and
        returned as a human-readable error string so tool failures
        never crash the agent loop.
        """
        entry = self.tools.get(name)
        if entry is None:
            return f"Error: tool '{name}' is not registered."

        try:
            result = entry["function"](**kwargs)
            # Ensure we always return a string
            if not isinstance(result, str):
                result = json.dumps(result, default=str)
            return result
        except Exception as exc:
            error_msg = f"Error executing tool '{name}': {type(exc).__name__}: {exc}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.list_tools()})"
