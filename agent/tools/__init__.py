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
    origin = getattr(type_hint, "__origin__", None)
    if origin is not None:
        args = getattr(type_hint, "__args__", ())
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

    def register(self, name: str, function: Callable, description: str, parameters: Dict[str, Any]) -> None:
        self.tools[name] = {"function": function, "description": description, "parameters": parameters}
        logger.debug("Registered tool: %s", name)

    def tool(self, name: str, description: str, parameters: Dict[str, Any]) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self.register(name, fn, description, parameters)
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return fn(*args, **kwargs)
            return wrapper
        return decorator

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        return self.tools.get(name)

    def get_openai_tools(self) -> List[Dict[str, Any]]:
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
        return list(self.tools.keys())

    def execute(self, name: str, **kwargs: Any) -> str:
        entry = self.tools.get(name)
        if entry is None:
            return f"Error: tool '{name}' is not registered."
        try:
            result = entry["function"](**kwargs)
            if not isinstance(result, str):
                result = json.dumps(result, default=str)
            return result
        except Exception as exc:
            error_msg = f"Error executing tool '{name}': {type(exc).__name__}: {exc}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def __len__(self) -> int:
        return len(self.tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.list_tools()})"
