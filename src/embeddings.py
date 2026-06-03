"""
Embedding pipeline: encodes opportunity text and persona profiles using
sentence-transformers, stores vectors in SQLite, builds a FAISS index.
"""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from database import (
    get_opportunities_with_embeddings,
    get_all_opportunities,
    save_embedding,
    save_persona_embedding,
    get_persona,
)

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode_text(text: str) -> list:
    model = get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def opportunity_text(row: dict) -> str:
    return f"{row['title']} {row.get('body', '')} {row.get('domain', '')} {row.get('source', '')}"


def embed_all_opportunities(batch_size: int = 64):
    """Compute and store embeddings for all opportunities that don't have one yet."""
    model = get_model()
    rows = get_all_opportunities()
    pending = [r for r in rows if r.get("embedding") is None]

    if not pending:
        print("All opportunities already embedded.")
        return

    print(f"Embedding {len(pending):,} opportunities...")
    texts = [opportunity_text(r) for r in pending]

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i: i + batch_size]
        batch_rows  = pending[i: i + batch_size]
        vecs = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
        for row, vec in zip(batch_rows, vecs):
            save_embedding(row["id"], vec.tolist())

    print("Embedding complete.")


def embed_persona(persona_name: str):
    """Encode a persona's interest profile and store it."""
    persona = get_persona(persona_name)
    if not persona:
        raise ValueError(f"Persona '{persona_name}' not found in DB.")
    vec = encode_text(persona["interests"])
    save_persona_embedding(persona_name, vec)
    return vec


# ── FAISS index ──────────────────────────────────────────────────────────────

def build_faiss_index():
    """Build an in-memory FAISS flat index from all embedded opportunities."""
    rows = get_opportunities_with_embeddings()
    if not rows:
        return None, []

    vecs = np.array([json.loads(r["embedding"]) for r in rows], dtype="float32")
    dim  = vecs.shape[1]

    index = faiss.IndexFlatIP(dim)   # inner product == cosine similarity (vecs are normalized)
    index.add(vecs)
    return index, rows


def search(query_vec: list, index, rows: list, top_k: int = 50) -> list:
    """Return top_k opportunities sorted by cosine similarity to query_vec."""
    q = np.array([query_vec], dtype="float32")
    distances, indices = index.search(q, min(top_k, len(rows)))
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        row = dict(rows[idx])
        row["similarity"] = float(dist)
        results.append(row)
    return results
