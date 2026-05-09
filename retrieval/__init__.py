# retrieval/__init__.py
from retrieval.embeddings import embed_single, embed_texts
from retrieval.vector_store import vector_store

__all__ = ["vector_store", "embed_texts", "embed_single"]
