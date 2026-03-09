"""
RAG (Retrieval-Augmented Generation) Pipeline.

Chunks documents, embeds them with ``sentence-transformers``, stores
vectors in ChromaDB, and retrieves the most relevant passages for
a given question.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Ingest documents and answer questions with vector retrieval."""

    def __init__(
        self,
        collection_name: str = "agent_docs",
        persist_dir: str = "./rag_data",
    ):
        """
        Parameters
        ----------
        collection_name : str
            ChromaDB collection to store / query document chunks.
        persist_dir : str
            Directory for the ChromaDB on-disk database.
        """
        self.collection_name = collection_name
        self.persist_dir = persist_dir

        # -- lazy imports so the module is loadable even without deps --
        try:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._chroma_client.get_or_create_collection(
                name=collection_name,
            )
            logger.info(
                "ChromaDB ready: collection='%s' persist='%s'",
                collection_name, persist_dir,
            )
        except ImportError:
            logger.error(
                "chromadb is not installed. Run: pip install chromadb"
            )
            self._chroma_client = None
            self._collection = None

        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("SentenceTransformer loaded: all-MiniLM-L6-v2")
        except ImportError:
            logger.error(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
            self._embedder = None

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> List[str]:
        """
        Split *text* into chunks of approximately *chunk_size* characters.

        Strategy
        --------
        1. Split the text on ``". "`` to get sentence-ish units.
        2. Accumulate sentences until *chunk_size* is reached.
        3. Overlap the last *overlap* characters into the next chunk.

        Returns
        -------
        list[str]
            Non-empty chunks.
        """
        sentences = text.split(". ")
        chunks: List[str] = []
        current = ""

        for sentence in sentences:
            # Re-add the period we split on (unless it's the last fragment)
            candidate = sentence if not current else f"{current}. {sentence}"

            if len(candidate) > chunk_size and current:
                # Flush the current chunk
                chunks.append(current.strip())
                # Start next chunk with overlap from the end of the previous
                overlap_text = current[-overlap:] if overlap else ""
                current = f"{overlap_text}{sentence}" if overlap_text else sentence
            else:
                current = candidate

        # Flush remaining
        if current.strip():
            chunks.append(current.strip())

        return chunks

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def load_file(self, file_path: str) -> str:
        """
        Read a file and return its text content.

        Supports ``.pdf`` (via PyPDF2), ``.md``, ``.txt``, and common
        source-code extensions.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(path))
                pages = [page.extract_text() or "" for page in reader.pages]
                text = "\n\n".join(pages)
                logger.info("Loaded PDF: %s (%d pages)", file_path, len(reader.pages))
                return text
            except ImportError:
                raise ImportError(
                    "PyPDF2 is required for PDF ingestion. "
                    "Run: pip install PyPDF2"
                )

        # Plain-text / source code
        content = path.read_text(encoding="utf-8")

        if ext in (".py", ".js", ".ts", ".json", ".yaml", ".yml"):
            return f"Source code ({ext}):\n{content}"

        return content

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings; returns list of float vectors."""
        if self._embedder is None:
            raise RuntimeError("SentenceTransformer not available.")
        return self._embedder.encode(texts, show_progress_bar=False).tolist()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Load a file, chunk it, embed the chunks, and add to ChromaDB.

        Returns the number of chunks ingested.
        """
        text = self.load_file(file_path)
        return self.ingest_text(text, source=file_path, metadata=metadata)

    def ingest_text(
        self,
        text: str,
        source: str = "direct_input",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Chunk raw text, embed, and store in ChromaDB.

        Returns the number of chunks ingested.
        """
        if self._collection is None:
            raise RuntimeError("ChromaDB collection not initialised.")

        chunks = self.chunk_text(text)
        if not chunks:
            logger.warning("No chunks produced from text (source=%s).", source)
            return 0

        embeddings = self._embed(chunks)

        # Build per-chunk metadata
        base_meta = metadata or {}
        ids: List[str] = []
        metas: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{source}::chunk_{i}"
            ids.append(chunk_id)
            meta = {"source": source, "chunk_index": i, **base_meta}
            metas.append(meta)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metas,
        )

        logger.info("Ingested %d chunks from '%s'.", len(chunks), source)
        return len(chunks)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for *question*.

        Returns
        -------
        list[dict]
            ``[{"text": str, "source": str, "score": float, "metadata": dict}, ...]``
            Sorted by relevance (lowest distance = best match).
        """
        if self._collection is None:
            raise RuntimeError("ChromaDB collection not initialised.")

        query_embedding = self._embed([question])

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )

        output: List[Dict[str, Any]] = []
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for doc, dist, meta in zip(documents, distances, metadatas):
            output.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "score": dist,
                "metadata": meta,
            })

        return output

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def list_sources(self) -> List[str]:
        """Return unique source paths stored in the collection."""
        if self._collection is None:
            return []

        all_meta = self._collection.get(include=["metadatas"])
        sources: set[str] = set()
        for meta in (all_meta.get("metadatas") or []):
            src = meta.get("source")
            if src:
                sources.add(src)
        return sorted(sources)

    def clear(self) -> None:
        """Delete and recreate the collection (wipes all data)."""
        if self._chroma_client is None:
            logger.warning("ChromaDB client not available; nothing to clear.")
            return

        self._chroma_client.delete_collection(self.collection_name)
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
        )
        logger.info("Collection '%s' cleared and recreated.", self.collection_name)

    def __repr__(self) -> str:
        count = 0
        if self._collection is not None:
            count = self._collection.count()
        return (
            f"RAGPipeline(collection='{self.collection_name}', "
            f"chunks={count})"
        )
