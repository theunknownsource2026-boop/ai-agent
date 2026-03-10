"""
Persistent Memory System with SQLite + ChromaDB.

Provides:
- SQLite for persistent storage (facts, chat threads, messages, summaries)
- ChromaDB for semantic similarity search (optional)
- Conversation history with named threads
- Auto-summarization of old messages
- Fact extraction and retrieval
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    fact TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    source TEXT DEFAULT 'user',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    is_active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_calls_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (thread_id) REFERENCES threads(id)
);

CREATE TABLE IF NOT EXISTS summaries (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    message_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (thread_id) REFERENCES threads(id)
);

CREATE TABLE IF NOT EXISTS ingested_files (
    path TEXT PRIMARY KEY,
    hash TEXT,
    ingested_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_threads_active ON threads(is_active);
"""


def _get_db(db_path: str) -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure schema exists."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.utcnow().isoformat()


def _uid() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Main Memory Class
# ---------------------------------------------------------------------------

class PersistentMemory:
    """
    Unified memory: SQLite for persistence, ChromaDB for semantic search.

    Handles:
    - Chat message history with named threads
    - Long-term fact storage and retrieval
    - Conversation summarization
    - Context building (memory-first approach)
    """

    def __init__(
        self,
        db_path: str = None,
        max_history: int = 50,
        use_embeddings: bool = False,
        persist_dir: str = None,
    ):
        self.db_path = db_path or config.DB_PATH
        self.max_history = max_history
        self.use_embeddings = use_embeddings

        # SQLite connection
        self._db = _get_db(self.db_path)

        # Current thread state
        self._current_thread_id: Optional[str] = None
        self._system_message: Optional[str] = None
        self._message_cache: List[Dict[str, str]] = []

        # ChromaDB (optional, for semantic search)
        self._collection = None
        self._embedder = None
        if use_embeddings:
            self._init_embeddings(persist_dir)

        # Auto-resume last active thread or create new one
        self._resume_or_create_thread()

        logger.info(
            "PersistentMemory initialized (db=%s, thread=%s, embeddings=%s)",
            self.db_path, self._current_thread_id, use_embeddings,
        )

    # ------------------------------------------------------------------
    # Embedding init (ChromaDB)
    # ------------------------------------------------------------------

    def _init_embeddings(self, persist_dir: str = None):
        """Initialize ChromaDB + SentenceTransformer for semantic search."""
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            persist_dir = persist_dir or str(
                Path(self.db_path).parent / "chroma_db"
            )
            client = chromadb.PersistentClient(path=persist_dir)
            self._collection = client.get_or_create_collection(
                name="agent_memory",
                metadata={"hnsw:space": "cosine"},
            )
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("ChromaDB initialized at %s", persist_dir)
        except ImportError:
            logger.warning(
                "chromadb or sentence-transformers not installed. "
                "Semantic search disabled."
            )
            self.use_embeddings = False
        except Exception as e:
            logger.warning("Could not init embeddings: %s", e)
            self.use_embeddings = False

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def _resume_or_create_thread(self):
        """Resume last active thread or create a fresh one."""
        row = self._db.execute(
            "SELECT id, name FROM threads WHERE is_active = 1 "
            "ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()

        if row:
            self._current_thread_id = row["id"]
            self._load_thread_messages()
            logger.info("Resumed thread: %s", row["name"])
        else:
            self.new_thread("default")

    def new_thread(self, name: str = None) -> str:
        """Create a new conversation thread and switch to it."""
        thread_id = _uid()
        name = name or f"chat-{thread_id[:6]}"
        now = _now()

        self._db.execute(
            "UPDATE threads SET is_active = 0 WHERE is_active = 1"
        )
        self._db.execute(
            "INSERT INTO threads (id, name, created_at, updated_at, is_active) "
            "VALUES (?, ?, ?, ?, 1)",
            (thread_id, name, now, now),
        )
        self._db.commit()

        self._current_thread_id = thread_id
        self._message_cache = []
        logger.info("Created new thread: %s (%s)", name, thread_id)
        return thread_id

    def switch_thread(self, thread_id: str) -> bool:
        """Switch to an existing thread by ID."""
        row = self._db.execute(
            "SELECT id FROM threads WHERE id = ?", (thread_id,)
        ).fetchone()
        if not row:
            return False

        self._db.execute("UPDATE threads SET is_active = 0")
        self._db.execute(
            "UPDATE threads SET is_active = 1, updated_at = ? WHERE id = ?",
            (_now(), thread_id),
        )
        self._db.commit()

        self._current_thread_id = thread_id
        self._load_thread_messages()
        return True

    def rename_thread(self, thread_id: str, new_name: str):
        """Rename an existing thread."""
        self._db.execute(
            "UPDATE threads SET name = ?, updated_at = ? WHERE id = ?",
            (new_name, _now(), thread_id),
        )
        self._db.commit()
        logger.info("Renamed thread %s to %s", thread_id, new_name)

    def delete_thread(self, thread_id: str):
        """Delete a thread and all its messages/summaries."""
        self._db.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
        self._db.execute("DELETE FROM summaries WHERE thread_id = ?", (thread_id,))
        self._db.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        self._db.commit()
        logger.info("Deleted thread %s", thread_id)

    def list_threads(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent conversation threads."""
        rows = self._db.execute(
            "SELECT id, name, created_at, updated_at, is_active "
            "FROM threads ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def _load_thread_messages(self):
        """Load messages for the current thread into cache."""
        rows = self._db.execute(
            "SELECT role, content, tool_calls_json FROM messages "
            "WHERE thread_id = ? ORDER BY created_at ASC",
            (self._current_thread_id,),
        ).fetchall()

        self._message_cache = []
        for r in rows:
            msg = {"role": r["role"], "content": r["content"]}
            if r["tool_calls_json"]:
                try:
                    msg["tool_calls"] = json.loads(r["tool_calls_json"])
                except json.JSONDecodeError:
                    pass
            self._message_cache.append(msg)

    # ------------------------------------------------------------------
    # Message management (backwards-compatible API)
    # ------------------------------------------------------------------

    def set_system(self, prompt: str):
        """Set the system message."""
        self._system_message = prompt

    def add_message(self, role: str, content: str, tool_calls: list = None):
        """Add a message to the current thread (memory + SQLite)."""
        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls

        self._message_cache.append(msg)

        # Persist to SQLite
        tc_json = json.dumps(tool_calls) if tool_calls else None
        now = _now()
        self._db.execute(
            "INSERT INTO messages (id, thread_id, role, content, tool_calls_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (_uid(), self._current_thread_id, role, content, tc_json, now),
        )
        self._db.execute(
            "UPDATE threads SET updated_at = ? WHERE id = ?",
            (now, self._current_thread_id),
        )
        self._db.commit()

        # Auto-summarize if cache is getting large
        if len(self._message_cache) > self.max_history:
            self._auto_summarize()

    def get_messages(self) -> List[Dict[str, str]]:
        """Get the message list for API calls (system + history)."""
        messages = []
        if self._system_message:
            messages.append({"role": "system", "content": self._system_message})

        # If we have summaries, prepend them as context
        summaries = self._get_thread_summaries()
        if summaries:
            summary_text = "Previous conversation context:\n" + "\n".join(summaries)
            messages.append({"role": "system", "content": summary_text})

        # Add recent messages (sliding window)
        recent = self._message_cache[-self.max_history:]
        messages.extend(recent)
        return messages

    def clear(self):
        """Clear the current conversation (start fresh thread)."""
        self._message_cache = []
        self.new_thread()

    # ------------------------------------------------------------------
    # Long-term fact memory
    # ------------------------------------------------------------------

    def remember(self, fact: str, category: str = "general", source: str = "user"):
        """Store a fact in long-term memory (SQLite + ChromaDB)."""
        fact_id = _uid()
        now = _now()

        # SQLite
        self._db.execute(
            "INSERT OR REPLACE INTO facts (id, fact, category, source, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (fact_id, fact, category, source, now),
        )
        self._db.commit()

        # ChromaDB (if available)
        if self._collection is not None and self._embedder is not None:
            try:
                embedding = self._embedder.encode(
                    [fact], show_progress_bar=False
                ).tolist()
                self._collection.add(
                    ids=[fact_id],
                    documents=[fact],
                    embeddings=embedding,
                    metadatas=[{
                        "category": category,
                        "source": source,
                        "timestamp": now,
                    }],
                )
            except Exception as e:
                logger.warning("ChromaDB add failed: %s", e)

        logger.info("Remembered: [%s] %s", category, fact[:80])

    def recall(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search long-term memory. Uses embeddings if available, else keywords."""
        if self.use_embeddings and self._collection and self._embedder and query:
            return self._recall_semantic(query, category, limit)
        return self._recall_sqlite(query, category, limit)

    def _recall_semantic(self, query: str, category: str = None, limit: int = 5) -> List[Dict]:
        """Semantic search via ChromaDB."""
        where_filter = {"category": category} if category else None
        embedding = self._embedder.encode([query], show_progress_bar=False).tolist()

        results = self._collection.query(
            query_embeddings=embedding,
            n_results=limit,
            where=where_filter,
        )

        documents = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        return [
            {
                "fact": doc,
                "category": meta.get("category", "general"),
                "timestamp": meta.get("timestamp", ""),
                "score": dist,
            }
            for doc, dist, meta in zip(documents, distances, metadatas)
        ]

    def _recall_sqlite(self, query: str = None, category: str = None, limit: int = 5) -> List[Dict]:
        """Keyword search via SQLite."""
        sql = "SELECT fact, category, source, created_at FROM facts WHERE 1=1"
        params: list = []

        if category:
            sql += " AND category = ?"
            params.append(category)

        if query:
            keywords = query.lower().split()
            for kw in keywords:
                sql += " AND LOWER(fact) LIKE ?"
                params.append(f"%{kw}%")

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._db.execute(sql, params).fetchall()
        return [
            {
                "fact": r["fact"],
                "category": r["category"],
                "timestamp": r["created_at"],
                "score": 0.0,
            }
            for r in rows
        ]

    def forget(self, query: str) -> int:
        """Delete facts matching a keyword query. Returns count deleted."""
        keywords = query.lower().split()
        sql = "SELECT id, fact FROM facts WHERE 1=1"
        params: list = []
        for kw in keywords:
            sql += " AND LOWER(fact) LIKE ?"
            params.append(f"%{kw}%")

        rows = self._db.execute(sql, params).fetchall()
        if not rows:
            return 0

        ids = [r["id"] for r in rows]
        placeholders = ",".join("?" for _ in ids)
        self._db.execute(f"DELETE FROM facts WHERE id IN ({placeholders})", ids)
        self._db.commit()

        # Also remove from ChromaDB if available
        if self._collection:
            try:
                self._collection.delete(ids=ids)
            except Exception as e:
                logger.warning("ChromaDB delete failed: %s", e)

        logger.info("Forgot %d fact(s) matching '%s'", len(ids), query)
        return len(ids)

    # ------------------------------------------------------------------
    # Context building (the KEY feature -- memory-first approach)
    # ------------------------------------------------------------------

    def get_relevant_context(self, query: str, limit: int = 5) -> str:
        """
        Build relevant context from ALL memory sources.

        This is what makes the agent 'already know' instead of
        saying 'let me go check'. Searches:
        1. Stored facts (semantic or keyword)
        2. Conversation summaries from past threads
        3. Recent messages from current thread

        Returns a formatted string to inject into the system prompt.
        """
        context_parts: list[str] = []

        # 1. Search stored facts
        facts = self.recall(query=query, limit=limit)
        if facts:
            fact_lines = [f"- {f['fact']}" for f in facts]
            context_parts.append(
                "REMEMBERED FACTS:\n" + "\n".join(fact_lines)
            )

        # 2. Search past conversation summaries
        summaries = self._search_all_summaries(query, limit=3)
        if summaries:
            context_parts.append(
                "PAST CONVERSATIONS:\n" + "\n".join(f"- {s}" for s in summaries)
            )

        if not context_parts:
            return ""

        return "\n\n".join(context_parts)

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _auto_summarize(self):
        """Compress older messages into a summary to save context window."""
        if len(self._message_cache) <= self.max_history:
            return

        # Take the oldest half of messages and summarize them
        split_point = len(self._message_cache) // 2
        old_messages = self._message_cache[:split_point]
        self._message_cache = self._message_cache[split_point:]

        # Build a simple summary from the old messages
        summary_parts = []
        for msg in old_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            if content.strip():
                summary_parts.append(f"{role}: {content}")

        if summary_parts:
            summary_text = "Conversation summary: " + " | ".join(summary_parts[:10])

            self._db.execute(
                "INSERT INTO summaries (id, thread_id, summary, message_count, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (_uid(), self._current_thread_id, summary_text, len(old_messages), _now()),
            )
            self._db.commit()
            logger.info("Auto-summarized %d messages", len(old_messages))

    def _get_thread_summaries(self) -> List[str]:
        """Get summaries for the current thread."""
        rows = self._db.execute(
            "SELECT summary FROM summaries WHERE thread_id = ? ORDER BY created_at ASC",
            (self._current_thread_id,),
        ).fetchall()
        return [r["summary"] for r in rows]

    def _search_all_summaries(self, query: str, limit: int = 3) -> List[str]:
        """Search summaries across ALL threads."""
        keywords = query.lower().split()[:5]
        sql = "SELECT summary FROM summaries WHERE 1=1"
        params: list = []
        for kw in keywords:
            sql += " AND LOWER(summary) LIKE ?"
            params.append(f"%{kw}%")
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._db.execute(sql, params).fetchall()
        return [r["summary"] for r in rows]

    # ------------------------------------------------------------------
    # Ingestion tracking (for auto-learning)
    # ------------------------------------------------------------------

    def is_file_ingested(self, file_path: str) -> bool:
        """Check if a file has already been ingested."""
        row = self._db.execute(
            "SELECT path FROM ingested_files WHERE path = ?", (file_path,)
        ).fetchone()
        return row is not None

    def mark_file_ingested(self, file_path: str, file_hash: str = ""):
        """Mark a file as ingested."""
        self._db.execute(
            "INSERT OR REPLACE INTO ingested_files (path, hash, ingested_at) VALUES (?, ?, ?)",
            (file_path, file_hash, _now()),
        )
        self._db.commit()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_memory_stats(self) -> Dict[str, Any]:
        """Return memory system statistics."""
        fact_count = self._db.execute("SELECT COUNT(*) as c FROM facts").fetchone()["c"]
        thread_count = self._db.execute("SELECT COUNT(*) as c FROM threads").fetchone()["c"]
        message_count = self._db.execute("SELECT COUNT(*) as c FROM messages").fetchone()["c"]
        summary_count = self._db.execute("SELECT COUNT(*) as c FROM summaries").fetchone()["c"]
        ingested_count = self._db.execute("SELECT COUNT(*) as c FROM ingested_files").fetchone()["c"]

        categories = {}
        for row in self._db.execute(
            "SELECT category, COUNT(*) as c FROM facts GROUP BY category"
        ).fetchall():
            categories[row["category"]] = row["c"]

        collection_size = 0
        if self._collection:
            try:
                collection_size = self._collection.count()
            except Exception:
                pass

        return {
            "facts": fact_count,
            "threads": thread_count,
            "messages": message_count,
            "summaries": summary_count,
            "ingested_files": ingested_count,
            "categories": categories,
            "chromadb_entries": collection_size,
            "backend": "sqlite+chromadb" if self.use_embeddings else "sqlite",
            "current_thread": self._current_thread_id,
        }

    def close(self):
        """Close the database connection."""
        if self._db:
            self._db.close()


# Backwards-compatible alias
ConversationMemory = PersistentMemory
