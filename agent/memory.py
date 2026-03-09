"""
Conversation memory with optional ChromaDB-backed long-term recall.

Maintains a sliding window of recent messages for context, and a
persistent store for facts the agent should remember across sessions.

When ``use_embeddings=True`` (the default), facts are embedded with
``sentence-transformers`` and stored in ChromaDB for semantic search.
The JSONL file is always kept as a reliable backup.  When embeddings
are disabled the module falls back to the original keyword-based search
for full backward compatibility.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent import config

logger = logging.getLogger(__name__)

# Default memory file lives in the project root
_DEFAULT_MEMORY_FILE = Path(__file__).resolve().parent.parent / "memory.jsonl"


class ConversationMemory:
    """Sliding-window message history + persistent long-term fact store."""

    def __init__(
        self,
        max_history: int = 20,
        memory_file: str | Path | None = None,
        use_embeddings: bool = True,
        persist_dir: str = "./memory_data",
    ):
        self.max_history = max_history
        self._memory_file = Path(memory_file) if memory_file else _DEFAULT_MEMORY_FILE
        self.messages: list[dict[str, str]] = []
        self.system_prompt: str = config.DEFAULT_SYSTEM_PROMPT
        self.use_embeddings = use_embeddings

        # Ensure JSONL backup file exists
        self._memory_file.touch(exist_ok=True)

        # ---- ChromaDB + embeddings (optional) ------------------------
        self._chroma_client = None
        self._collection = None
        self._embedder = None

        if use_embeddings:
            self._init_embeddings(persist_dir)

    # ------------------------------------------------------------------
    # Embedding initialisation (lazy, fault-tolerant)
    # ------------------------------------------------------------------

    def _init_embeddings(self, persist_dir: str) -> None:
        """Attempt to load ChromaDB + SentenceTransformer."""
        try:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._chroma_client.get_or_create_collection(
                name="long_term_memory",
            )
            logger.info(
                "Memory ChromaDB ready: collection='long_term_memory' persist='%s'",
                persist_dir,
            )
        except ImportError:
            logger.warning(
                "chromadb not installed — falling back to keyword search. "
                "Run: pip install chromadb"
            )
            self.use_embeddings = False
            return
        except Exception as exc:
            logger.warning("ChromaDB init failed (%s) — keyword fallback.", exc)
            self.use_embeddings = False
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("SentenceTransformer loaded for memory embeddings.")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — keyword fallback. "
                "Run: pip install sentence-transformers"
            )
            self.use_embeddings = False

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def set_system(self, prompt: str) -> None:
        """Set (or replace) the system prompt."""
        self.system_prompt = prompt

    # ------------------------------------------------------------------
    # Sliding-window message management
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """
        Append a message and trim the window to ``max_history``.

        The system message is prepended by :meth:`get_messages` and is
        **not** counted against the window size.
        """
        self.messages.append({"role": role, "content": content})

        # Trim oldest messages, keeping the most recent max_history
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def get_messages(self) -> list[dict[str, str]]:
        """Return the full message list with the system prompt prepended."""
        system_msg = {"role": "system", "content": self.system_prompt}
        return [system_msg] + list(self.messages)

    def clear(self) -> None:
        """Reset the sliding window (keeps the system prompt)."""
        self.messages.clear()

    # ------------------------------------------------------------------
    # Persistent fact store — remember
    # ------------------------------------------------------------------

    def remember(self, fact: str, category: str = "general") -> None:
        """
        Persist a fact to long-term memory.

        Always writes to the JSONL backup.  When embeddings are enabled
        the fact is also embedded and upserted into ChromaDB.

        Parameters
        ----------
        fact : str
            The piece of information to remember.
        category : str
            A tag for filtering later (e.g. ``"user_pref"``, ``"project"``).
        """
        timestamp = datetime.utcnow().isoformat() + "Z"

        # ---- Always write JSONL backup --------------------------------
        entry = {"fact": fact, "category": category, "timestamp": timestamp}
        try:
            with open(self._memory_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            logger.debug("Remembered (JSONL): %s [%s]", fact[:80], category)
        except OSError as exc:
            logger.error("Failed to write memory JSONL: %s", exc)

        # ---- ChromaDB embedding (if available) ------------------------
        if self.use_embeddings and self._collection is not None and self._embedder is not None:
            try:
                embedding = self._embedder.encode([fact], show_progress_bar=False).tolist()
                doc_id = f"fact_{timestamp}_{hash(fact) & 0xFFFFFFFF:08x}"
                self._collection.upsert(
                    ids=[doc_id],
                    embeddings=embedding,
                    documents=[fact],
                    metadatas=[{"category": category, "timestamp": timestamp}],
                )
                logger.debug("Remembered (ChromaDB): %s", doc_id)
            except Exception as exc:
                logger.error("ChromaDB upsert failed: %s", exc)

    # ------------------------------------------------------------------
    # Persistent fact store — recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict]:
        """
        Search long-term memory.

        When embeddings are available the query is embedded and
        ChromaDB returns results sorted by cosine similarity.
        Otherwise falls back to keyword matching against the JSONL file.

        Parameters
        ----------
        query : str, optional
            Natural-language query (semantic) or keywords (fallback).
        category : str, optional
            Only return facts with this category tag.
        limit : int
            Maximum results to return.

        Returns
        -------
        list[dict]
            Matching memory entries.  When using embeddings each dict
            has an extra ``"score"`` key (lower = more similar).
        """
        if self.use_embeddings and self._collection is not None and self._embedder is not None:
            return self._recall_embeddings(query, category, limit)
        return self._recall_keywords(query, category, limit)

    # ---- Embedding-based recall --------------------------------------

    def _recall_embeddings(
        self,
        query: Optional[str],
        category: Optional[str],
        limit: int,
    ) -> list[dict]:
        """Semantic recall via ChromaDB."""
        where_filter: Optional[Dict[str, Any]] = None
        if category:
            where_filter = {"category": category}

        if query:
            query_embedding = self._embedder.encode(
                [query], show_progress_bar=False,
            ).tolist()
            results = self._collection.query(
                query_embeddings=query_embedding,
                n_results=limit,
                where=where_filter,
            )
        else:
            # No query — return most recent entries
            results = self._collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
            )
            # Wrap into query-style structure for uniform processing
            docs = results.get("documents") or []
            metas = results.get("metadatas") or []
            # Sort by timestamp descending, take `limit`
            paired = list(zip(docs, metas))
            paired.sort(
                key=lambda x: x[1].get("timestamp", "") if x[1] else "",
                reverse=True,
            )
            paired = paired[:limit]
            return [
                {
                    "fact": doc,
                    "category": meta.get("category", "general"),
                    "timestamp": meta.get("timestamp", ""),
                    "score": 0.0,
                }
                for doc, meta in paired
            ]

        # Unpack query results
        documents = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        output: list[dict] = []
        for doc, dist, meta in zip(documents, distances, metadatas):
            output.append({
                "fact": doc,
                "category": meta.get("category", "general"),
                "timestamp": meta.get("timestamp", ""),
                "score": dist,
            })

        # Already sorted by similarity (lowest distance first)
        return output

    # ---- Keyword-based recall (original, backward-compatible) --------

    def _recall_keywords(
        self,
        query: Optional[str],
        category: Optional[str],
        limit: int,
    ) -> list[dict]:
        """Keyword-based recall from the JSONL file."""
        entries = self._load_memories()

        # Filter by category
        if category:
            entries = [e for e in entries if e.get("category") == category]

        # Keyword match (case-insensitive)
        if query:
            keywords = query.lower().split()
            scored: list[tuple[int, dict]] = []
            for entry in entries:
                fact_lower = entry.get("fact", "").lower()
                hits = sum(1 for kw in keywords if kw in fact_lower)
                if hits > 0:
                    scored.append((hits, entry))
            scored.sort(
                key=lambda x: (x[0], x[1].get("timestamp", "")),
                reverse=True,
            )
            entries = [e for _, e in scored]
        else:
            entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

        return entries[:limit]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_memories(self) -> list[dict]:
        """Load all entries from the JSONL memory file."""
        entries: list[dict] = []
        try:
            with open(self._memory_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass
        return entries

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Return statistics about stored long-term memories.

        Returns
        -------
        dict
            ``{"total": int, "categories": {cat: count, ...}, "collection_size": int, "backend": str}``
        """
        entries = self._load_memories()
        categories = Counter(e.get("category", "unknown") for e in entries)

        collection_size = 0
        if self._collection is not None:
            try:
                collection_size = self._collection.count()
            except Exception:
                pass

        return {
            "total": len(entries),
            "categories": dict(categories),
            "collection_size": collection_size,
            "backend": "chromadb" if self.use_embeddings else "jsonl_keywords",
        }

    # ------------------------------------------------------------------
    # Utilities (preserved from original)
    # ------------------------------------------------------------------

    def memory_count(self) -> int:
        """Return the total number of stored facts."""
        return len(self._load_memories())

    def clear_memories(self) -> None:
        """Erase all persistent memories. Use with caution."""
        with open(self._memory_file, "w", encoding="utf-8") as fh:
            fh.truncate(0)

        if self._collection is not None:
            try:
                # Delete and recreate collection
                if self._chroma_client is not None:
                    self._chroma_client.delete_collection("long_term_memory")
                    self._collection = self._chroma_client.get_or_create_collection(
                        name="long_term_memory",
                    )
            except Exception as exc:
                logger.error("Failed to clear ChromaDB collection: %s", exc)

        logger.info("All memories cleared.")

    def get_context_window_info(self) -> dict:
        """Return info about the current sliding window state."""
        return {
            "messages_count": len(self.messages),
            "max_history": self.max_history,
            "system_prompt_length": len(self.system_prompt),
            "total_chars": sum(len(m.get("content", "")) for m in self.messages),
        }

    def __repr__(self) -> str:
        backend = "chromadb" if self.use_embeddings else "jsonl"
        return (
            f"ConversationMemory(messages={len(self.messages)}/{self.max_history}, "
            f"stored_facts={self.memory_count()}, backend={backend})"
        )
