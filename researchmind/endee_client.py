"""
endee_client.py
Thin wrapper around the Endee Python SDK for ResearchMind.
"""

from __future__ import annotations
import openai
from endee import Endee, Precision


INDEX_NAME = "researchmind"
DIMENSION = 1536          # OpenAI text-embedding-3-small dimension
SPACE_TYPE = "cosine"


class EndeeVectorStore:
    """Manages a single Endee index for document storage and retrieval."""

    def __init__(self, base_url: str = "http://localhost:8080", auth_token: str | None = None):
        token = auth_token or ""
        self.client = Endee(token)
        self.client.set_base_url(f"{base_url.rstrip('/')}/api/v1")
        self.index = None

    # ── Setup ────────────────────────────────────────────────────────────────
    def ensure_index(self):
        """Create the index if it doesn't exist, then cache a reference."""
        existing = [idx.name for idx in self.client.list_indexes()]
        if INDEX_NAME not in existing:
            self.client.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                space_type=SPACE_TYPE,
                precision=Precision.INT8,
            )
        self.index = self.client.get_index(name=INDEX_NAME)

    # ── Ingest ───────────────────────────────────────────────────────────────
    def upsert_chunk(self, doc_id: str, text: str, meta: dict, openai_key: str):
        """Embed a text chunk with OpenAI and upsert into Endee."""
        if self.index is None:
            self.ensure_index()

        vector = self._embed(text, openai_key)
        meta["text"] = text          # store raw text in metadata for retrieval

        self.index.upsert([{
            "id": doc_id,
            "vector": vector,
            "meta": meta,
        }])

    # ── Query ────────────────────────────────────────────────────────────────
    def search(self, query: str, openai_key: str, top_k: int = 5) -> list[dict]:
        """Embed a query and return top-k nearest neighbours with metadata."""
        if self.index is None:
            self.ensure_index()

        vector = self._embed(query, openai_key)
        results = self.index.query(vector=vector, top_k=top_k)

        hits = []
        for r in results:
            hits.append({
                "id": r.id,
                "score": round(r.similarity, 4),
                "text": r.meta.get("text", ""),
                "source": r.meta.get("source", "unknown"),
                "chunk": r.meta.get("chunk", 0),
            })
        return hits

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _embed(text: str, openai_key: str) -> list[float]:
        """Generate an embedding using OpenAI text-embedding-3-small."""
        client = openai.OpenAI(api_key=openai_key)
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return resp.data[0].embedding
