from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()


class HashEmbeddingFunction:
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def name(self) -> str:
        return "hash"

    def __call__(self, input):
        return [self._embed(text) for text in input]

    def embed_query(self, input):
        if isinstance(input, str):
            return self._embed(input)
        return [self._embed(text) for text in input]

    def embed_documents(self, input):
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def get_embedding_function():
    provider = os.getenv("EMBEDDING_PROVIDER", "hash").lower()
    if provider == "hash":
        return HashEmbeddingFunction()
    if provider == "openai":
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    )


def get_collection():
    chroma_dir = Path(os.getenv("CHROMA_DIR", "vectorstore/chroma_db"))
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_or_create_collection(
        name="insurance_documents",
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def retrieve_documents(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    collection = get_collection()
    n_results = top_k or int(os.getenv("TOP_K", "6"))
    if collection.count() == 0:
        return []

    results = collection.query(query_texts=[question], n_results=n_results)
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    retrieved: list[dict[str, Any]] = []
    for index, content in enumerate(docs):
        distance = distances[index] if index < len(distances) else 1.0
        metadata = metadatas[index] if index < len(metadatas) else {}
        similarity = max(0.0, 1.0 - float(distance))
        boosted_similarity = similarity + _category_boost(question, metadata)
        retrieved.append(
            {
                "id": ids[index] if index < len(ids) else str(index),
                "content": content,
                "metadata": metadata,
                "distance": distance,
                "similarity": min(1.0, boosted_similarity),
            }
        )
    return sorted(retrieved, key=lambda doc: doc["similarity"], reverse=True)


def _category_boost(question: str, metadata: dict[str, Any]) -> float:
    category = str(metadata.get("category", "")).lower()
    lowered = question.lower()
    keywords = {
        "motor": ["motor", "car", "vehicle", "driver", "accident"],
        "health": ["health", "medical", "hospital", "patient", "treatment"],
        "home": ["home", "house", "property", "water", "storm", "theft"],
        "claims": ["claim", "denied", "appeal", "documents", "procedure"],
        "faq": ["faq", "exclusion", "cancel", "third-party"],
        "regulations": ["regulation", "regulatory", "renewal", "complaint"],
    }
    if any(word in lowered for word in keywords.get(category, [])):
        return 0.15
    return 0.0
