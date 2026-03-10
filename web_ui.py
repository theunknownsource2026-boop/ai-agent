#!/usr/bin/env python3
"""
Web UI for the Multi-Provider AI Agent.
Run:  python web_ui.py
Open: http://localhost:5000
"""

import sys, os, time, json, shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify

from agent import config
from agent.memory import PersistentMemory
from agent.budget import BudgetTracker
from agent.router import Router, build_providers
from agent.rag import RAGPipeline
from agent.tools.builtin import default_registry

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max upload

# Upload folder for RAG documents
UPLOAD_DIR = Path(__file__).parent / "rag_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Agent singleton
# ---------------------------------------------------------------------------
memory = PersistentMemory(use_embeddings=False)
memory.set_system(config.DEFAULT_SYSTEM_PROMPT)
budget = BudgetTracker()
tools = default_registry

# Build ALL providers dynamically from config registry
providers = build_providers()
router = Router(providers, budget)
rag = None  # lazy init

# Runtime settings
runtime = {
    "provider_override": None,
    "model_override": None,
    "temperature": 0.7,
    "max_tokens": 4096,
    "timeout": 120,
    "system_prompt": config.DEFAULT_SYSTEM_PROMPT,
    "uncensored": False,
}


def _get_rag():
    global rag
    if rag is None:
        rag = RAGPipeline()
    return rag


def _get_provider_status():
    """Dynamically detect all available providers and their models."""
    status = {}
    for name, prov in providers.items():
        try:
            is_avail = prov.is_available()
        except Exception:
            is_avail = False

        models = []
        if is_avail:
            # Try to get models from config registry
            reg = config.PROVIDER_REGISTRY.get(name, {})
            model_list = reg.get("models", {})
            if model_list:
                models = list(model_list.keys()) if isinstance(model_list, dict) else list(model_list)
            elif name == "ollama":
                try:
                    models = prov.list_models()
                except Exception:
                    models = [getattr(config, "OLLAMA_DOLPHIN", "dolphin-llama3")]

        status[name] = {"available": is_avail, "models": models}
    return status


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/status")
def api_status():
    r = _get_rag()
    try:
        rag_sources = r.list_sources() if r._collection else []
    except Exception:
        rag_sources = []
    try:
        rag_count = r._collection.count() if r._collection else 0
    except Exception:
        rag_count = 0

    mem_stats = memory.get_memory_stats()

    return jsonify({
        "providers": _get_provider_status(),
        "settings": runtime,
        "budget": {
            "daily_used": budget._today_cost(),
            "daily_limit": config.DAILY_BUDGET_LIMIT,
            "monthly_used": budget._month_cost(),
            "monthly_limit": config.MONTHLY_BUDGET_LIMIT,
        },
        "rag": {
            "sources": rag_sources,
            "total_chunks": rag_count,
        },
        "memory": mem_stats,
    })


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.json
    if "provider_override" in data:
        val = data["provider_override"]
        runtime["provider_override"] = val if val != "auto" else None
    if "model_override" in data:
        val = data["model_override"]
        runtime["model_override"] = val if val else None
    if "temperature" in data:
        runtime["temperature"] = max(0.0, min(2.0, float(data["temperature"])))
    if "max_tokens" in data:
        runtime["max_tokens"] = max(64, min(32768, int(data["max_tokens"])))
    if "timeout" in data:
        runtime["timeout"] = max(10, min(600, int(data["timeout"])))
        if "ollama" in providers:
            providers["ollama"]._timeout = runtime["timeout"]
    if "system_prompt" in data:
        runtime["system_prompt"] = data["system_prompt"]
        memory.set_system(data["system_prompt"])
    if "uncensored" in data:
        runtime["uncensored"] = bool(data["uncensored"])
    return jsonify({"ok": True, "settings": runtime})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    memory.add_message("user", user_msg)

    # Resolve provider
    try:
        if runtime["provider_override"]:
            prov_name = runtime["provider_override"]
            provider = providers.get(prov_name)
            model = runtime["model_override"] or getattr(provider, "_default_model", None)
            if not provider or not provider.is_available():
                return jsonify({"error": f"Provider {prov_name} not available"}), 503
            route_info = {"intent": "manual", "provider": prov_name, "model": model}
        elif runtime["uncensored"]:
            provider = providers.get("ollama")
            model = getattr(config, "OLLAMA_DOLPHIN", "dolphin-llama3")
            prov_name = "ollama"
            route_info = {"intent": "uncensored", "provider": "ollama", "model": model}
            if not provider or not provider.is_available():
                return jsonify({"error": "Ollama not available"}), 503
        else:
            provider, model, route_info = router.route(user_msg)
            prov_name = route_info["provider"]
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503

    # -----------------------------------------------------------
    # Context injection: memory-first approach
    # -----------------------------------------------------------

    # 1. Long-term memory context (facts + past conversation summaries)
    mem_context = memory.get_relevant_context(user_msg, limit=5)

    # 2. RAG context (ingested documents)
    rag_context = ""
    if rag and rag._collection:
        try:
            count = rag._collection.count()
        except Exception:
            count = 0
        if count > 0:
            try:
                results = rag.query(user_msg, n_results=3)
                if results:
                    rag_parts = []
                    for r in results:
                        rag_parts.append(f"[Source: {r['source']}]\n{r['text'][:500]}")
                    rag_context = "RELEVANT DOCUMENTS:\n" + "\n\n".join(rag_parts)
            except Exception:
                pass

    # Build messages with context injected
    messages = memory.get_messages()

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

    # Get tool schemas for providers that support function calling
    tools_schema = None
    prov_obj = providers.get(prov_name)
    if prov_obj and getattr(prov_obj, '_supports_tools', True):
        try:
            tools_schema = tools.get_openai_tools()
        except Exception:
            tools_schema = None

    # Call provider with tool loop
    max_rounds = 5
    elapsed = 0
    for round_num in range(max_rounds):
        try:
            t0 = time.time()
            kwargs = {
                "messages": messages,
                "model": model,
                "temperature": runtime["temperature"],
            }
            if tools_schema:
                kwargs["tools"] = tools_schema
            response = provider.chat(**kwargs)
            elapsed = time.time() - t0
        except Exception as e:
            # Try fallback
            try:
                provider, model, route_info = router.route(user_msg)
                prov_name = route_info["provider"]
                kwargs = {
                    "messages": messages,
                    "model": model,
                    "temperature": runtime["temperature"],
                }
                if tools_schema:
                    kwargs["tools"] = tools_schema
                t0 = time.time()
                response = provider.chat(**kwargs)
                elapsed = time.time() - t0
            except Exception as e2:
                return jsonify({"error": f"All providers failed: {e2}"}), 500

        budget.log_call(
            model=response.model,
            provider=response.provider,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        # Handle tool calls -- normalize with type: function
        if response.tool_calls:
            normalized_tcs = []
            for tc in response.tool_calls:
                if isinstance(tc, dict):
                    ntc = dict(tc)
                else:
                    ntc = {
                        "id": getattr(tc, "id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.function.name if hasattr(tc, "function") else "",
                            "arguments": tc.function.arguments if hasattr(tc, "function") else "",
                        },
                    }
                ntc.setdefault("type", "function")
                normalized_tcs.append(ntc)

            messages.append({
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": normalized_tcs,
            })

            for tc in normalized_tcs:
                func_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    args = {}
                try:
                    result = tools.execute(func_name, **args)
                    # Wire remember_fact to actual memory
                    if func_name == "remember_fact":
                        fact = args.get("fact", "")
                        category = args.get("category", "general")
                        if fact:
                            memory.remember(fact, category)
                except Exception as ex:
                    result = f"Tool error: {ex}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", func_name),
                    "content": str(result),
                })
            continue

        # Final response
        if response.content:
            memory.add_message("assistant", response.content)

        return jsonify({
            "content": response.content or "",
            "provider": prov_name,
            "model": model,
            "intent": route_info.get("intent", "chat"),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "elapsed": round(elapsed, 2),
            "rag_used": bool(rag_context),
            "memory_used": bool(mem_context),
            "thread_id": memory._current_thread_id,
        })

    return jsonify({"error": "Max tool rounds exceeded"}), 500


# ---------------------------------------------------------------------------
# Thread management API
# ---------------------------------------------------------------------------

@app.route("/api/threads")
def api_threads():
    """List all conversation threads."""
    threads = memory.list_threads(limit=50)
    return jsonify({"threads": threads, "current": memory._current_thread_id})


@app.route("/api/threads/new", methods=["POST"])
def api_thread_new():
    """Create a new conversation thread."""
    data = request.json or {}
    name = data.get("name", None)
    tid = memory.new_thread(name)
    memory.set_system(runtime["system_prompt"])
    return jsonify({"ok": True, "thread_id": tid})


@app.route("/api/threads/switch", methods=["POST"])
def api_thread_switch():
    """Switch to an existing thread."""
    data = request.json or {}
    thread_id = data.get("thread_id", "")
    if not thread_id:
        return jsonify({"error": "thread_id required"}), 400

    # Support partial ID or name match
    threads = memory.list_threads()
    match = None
    for t in threads:
        if t["id"].startswith(thread_id) or t["name"] == thread_id:
            match = t["id"]
            break

    if match and memory.switch_thread(match):
        memory.set_system(runtime["system_prompt"])
        return jsonify({"ok": True, "thread_id": match})
    return jsonify({"error": f"Thread not found: {thread_id}"}), 404


@app.route("/api/threads/rename", methods=["POST"])
def api_thread_rename():
    """Rename a conversation thread."""
    data = request.json or {}
    thread_id = data.get("thread_id", memory._current_thread_id)
    new_name = data.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "name required"}), 400
    memory.rename_thread(thread_id, new_name)
    return jsonify({"ok": True})


@app.route("/api/threads/delete", methods=["POST"])
def api_thread_delete():
    """Delete a conversation thread."""
    data = request.json or {}
    thread_id = data.get("thread_id", "")
    if not thread_id:
        return jsonify({"error": "thread_id required"}), 400
    if thread_id == memory._current_thread_id:
        return jsonify({"error": "Cannot delete active thread. Switch first."}), 400
    memory.delete_thread(thread_id)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Memory API
# ---------------------------------------------------------------------------

@app.route("/api/memory/stats")
def api_memory_stats():
    """Get memory system statistics."""
    return jsonify(memory.get_memory_stats())


@app.route("/api/memory/remember", methods=["POST"])
def api_remember():
    data = request.json
    fact = data.get("fact", "").strip()
    category = data.get("category", "general")
    if fact:
        memory.remember(fact, category)
        return jsonify({"ok": True})
    return jsonify({"error": "Empty fact"}), 400


@app.route("/api/memory/recall")
def api_recall():
    query = request.args.get("q", "")
    category = request.args.get("category", None)
    limit = int(request.args.get("limit", 10))
    memories = memory.recall(query=query or None, category=category, limit=limit)
    return jsonify({"memories": memories})


@app.route("/api/memory/forget", methods=["POST"])
def api_forget():
    """Delete facts matching a query."""
    data = request.json or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query required"}), 400
    deleted = memory.forget(query)
    return jsonify({"ok": True, "deleted": deleted})


@app.route("/api/memory/summary", methods=["POST"])
def api_summary():
    """Manually trigger conversation summarization."""
    cache_len = len(memory._message_cache)
    if cache_len < 4:
        return jsonify({"ok": False, "message": "Not enough messages to summarize"})
    memory._auto_summarize()
    new_len = len(memory._message_cache)
    return jsonify({
        "ok": True,
        "summarized": cache_len - new_len,
        "remaining": new_len,
    })


# ---------------------------------------------------------------------------
# RAG API
# ---------------------------------------------------------------------------

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Handle drag & drop file uploads for RAG ingestion."""
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    r = _get_rag()
    results = []
    files = request.files.getlist("files")

    for f in files:
        if not f.filename:
            continue

        # Sanitize filename
        safe_name = f.filename.replace("..", "").replace("/", "_").replace("\\", "_")
        save_path = UPLOAD_DIR / safe_name

        # Handle duplicates
        if save_path.exists():
            stem = save_path.stem
            suffix = save_path.suffix
            counter = 1
            while save_path.exists():
                save_path = UPLOAD_DIR / f"{stem}_{counter}{suffix}"
                counter += 1

        f.save(str(save_path))

        # Ingest into RAG
        try:
            num_chunks = r.ingest(str(save_path))
            memory.mark_file_ingested(str(save_path))
            results.append({
                "filename": safe_name,
                "chunks": num_chunks,
                "status": "ok",
                "path": str(save_path),
            })
        except Exception as e:
            results.append({
                "filename": safe_name,
                "chunks": 0,
                "status": f"error: {e}",
            })

    return jsonify({"results": results})


@app.route("/api/rag/sources")
def api_rag_sources():
    r = _get_rag()
    try:
        sources = r.list_sources() if r._collection else []
    except Exception:
        sources = []
    try:
        count = r._collection.count() if r._collection else 0
    except Exception:
        count = 0
    return jsonify({"sources": sources, "total_chunks": count})


@app.route("/api/rag/clear", methods=["POST"])
def api_rag_clear():
    r = _get_rag()
    r.clear()
    # Also clean upload dir
    for f in UPLOAD_DIR.iterdir():
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass
    return jsonify({"ok": True})


@app.route("/api/clear", methods=["POST"])
def api_clear():
    memory.clear()
    memory.set_system(runtime["system_prompt"])
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("\n  AI Agent Control Panel")
    print(f"  Providers: {', '.join(providers.keys())}")
    print(f"  Memory: {memory.get_memory_stats()['facts']} facts, {memory.get_memory_stats()['threads']} threads")
    print("  Open: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
