"""
embeddings.py — Turns text into vectors and builds a searchable index.

WHY DO WE NEED EMBEDDINGS?
---------------------------
A database of 14,000 records can't be searched by keyword alone — a persona
interested in "neural networks" should also see results about "deep learning"
or "transformer models" even if those exact words aren't in their profile.

Embeddings solve this: we convert every piece of text (repo title, post body,
persona interests) into a list of ~384 numbers (a "vector") that captures
*meaning*, not just words. Two texts with similar meaning end up with similar
vectors — so we can measure relevance mathematically.

PIPELINE OVERVIEW:
  1. encode_text()         — convert any string → vector (384 floats)
  2. embed_all_opportunities() — encode every record in the DB and store the vector
  3. embed_persona()       — encode a persona's interest profile
  4. build_faiss_index()   — load all stored vectors into FAISS for fast search
  5. search()              — given a persona vector, find the closest opportunity vectors
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

# ── Model ─────────────────────────────────────────────────────────────────────

# "all-MiniLM-L6-v2" is a lightweight but powerful sentence embedding model.
# It produces 384-dimensional vectors and runs fast on CPU.
# "MiniLM" = distilled from a larger model; "L6" = 6 transformer layers.
# normalize_embeddings=True means every vector has length 1 (unit norm),
# which lets us use dot product as a direct measure of cosine similarity.
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None   # loaded lazily — only when first needed, to avoid slow imports at startup


def get_model() -> SentenceTransformer:
    """
    Lazy-load the sentence transformer model.
    Using a global variable avoids reloading the model on every function call
    (loading takes ~2s and downloads ~90MB the first time).
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ── Text encoding ─────────────────────────────────────────────────────────────

def encode_text(text: str) -> list:
    """
    Convert a plain text string into a normalised embedding vector.

    Steps:
      1. The model tokenises the text into subwords.
      2. A transformer network processes the tokens and produces a 384-dim vector.
      3. normalize_embeddings=True divides the vector by its own length (L2 norm),
         so the result always has magnitude = 1.0.
         This is important because it makes dot product == cosine similarity.

    Returns: list of 384 floats (the embedding vector).
    """
    model = get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def opportunity_text(row: dict) -> str:
    """
    Combine the most informative fields of an opportunity into a single string
    that will be fed to the embedding model.

    We concatenate title + body + domain + source so the vector captures:
      - What the project/post is about (title + body)
      - The tech area it belongs to (domain)
      - Where it came from (source — subtle signal for context)
    """
    return f"{row['title']} {row.get('body', '')} {row.get('domain', '')} {row.get('source', '')}"


# ── Batch embedding ───────────────────────────────────────────────────────────

def embed_all_opportunities(batch_size: int = 64):
    """
    Encode every opportunity in the DB that doesn't yet have an embedding,
    and save the vector back to the 'embedding' column in SQLite.

    Why batch processing?
      Encoding one text at a time is slow. The transformer model is much more
      efficient when given a batch of texts simultaneously (GPU parallelism or
      batched CPU matrix ops). batch_size=64 is a good balance between speed
      and memory usage on a laptop CPU.

    Why only encode records without embeddings?
      The seed data (~14k records) has no embeddings — they're stripped to keep
      the .gz file small. This function fills them in progressively. It's safe
      to call multiple times; already-embedded records are skipped.
    """
    model = get_model()
    rows = get_all_opportunities()

    # Filter to only records missing an embedding
    pending = [r for r in rows if r.get("embedding") is None]

    if not pending:
        print("All opportunities already embedded.")
        return

    print(f"Embedding {len(pending):,} opportunities...")
    texts = [opportunity_text(r) for r in pending]

    # Process in batches; tqdm shows a progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i: i + batch_size]
        batch_rows  = pending[i: i + batch_size]

        # model.encode() returns a 2D numpy array: shape (batch_size, 384)
        vecs = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)

        # Save each vector as a JSON-encoded list in SQLite
        for row, vec in zip(batch_rows, vecs):
            save_embedding(row["id"], vec.tolist())

    print("Embedding complete.")


def embed_persona(persona_name: str):
    """
    Encode a persona's interest description and store the resulting vector.

    Example: "Python, data engineering, cloud APIs, Apache Spark"
    → a 384-dim vector that will be used as the query when searching for
      relevant opportunities for this persona.

    The persona vector is stored in the 'personas' table so we don't
    re-encode it on every page load.
    """
    persona = get_persona(persona_name)
    if not persona:
        raise ValueError(f"Persona '{persona_name}' not found in DB.")
    vec = encode_text(persona["interests"])
    save_persona_embedding(persona_name, vec)
    return vec


# ── FAISS index ───────────────────────────────────────────────────────────────

def build_faiss_index():
    """
    Load all stored opportunity vectors from SQLite into a FAISS index for
    fast nearest-neighbour search.

    WHAT IS FAISS?
      FAISS (Facebook AI Similarity Search) is a library for efficient
      similarity search over large sets of vectors. Instead of comparing a
      query vector against every record one-by-one (O(n)), FAISS uses
      optimised data structures to find the closest vectors much faster.

    WHICH INDEX TYPE?
      We use IndexFlatIP — "Flat" means it's an exact (brute-force) search,
      "IP" means Inner Product (dot product). Because all our vectors are
      L2-normalised, inner product == cosine similarity. Exact search is fine
      at 14k records; approximate indexes (like IVF) are only needed at millions.

    Returns:
      index — the FAISS index object (in memory)
      rows  — list of dicts, in the same order as the index; row[i] corresponds
              to the i-th vector in the index
    """
    rows = get_opportunities_with_embeddings()
    if not rows:
        return None, []

    # Stack all vectors into a 2D float32 array: shape (n_records, 384)
    # FAISS requires float32 specifically
    vecs = np.array([json.loads(r["embedding"]) for r in rows], dtype="float32")
    dim  = vecs.shape[1]   # should be 384

    # Build the index and add all vectors at once
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    return index, rows


# ── Search ────────────────────────────────────────────────────────────────────

def search(query_vec: list, index, rows: list, top_k: int = 50) -> list:
    """
    Find the top_k most similar opportunities to a given query vector.

    HOW IT WORKS:
      1. Convert the query vector to a float32 numpy array (shape: 1 × 384).
      2. Ask FAISS to return the top_k nearest vectors by inner product.
      3. FAISS returns two arrays:
           distances[0] — the similarity scores (higher = more similar)
           indices[0]   — the row positions in the index
      4. We look up the original row metadata using those indices.

    Why top_k=50 by default?
      We retrieve more candidates than we'll show (15) so the downstream
      composite scorer and UCB bandit have enough material to re-rank.
      Fetching 50 gives ranking headroom without being too slow.

    Returns:
      List of opportunity dicts, each with an added 'similarity' key (0–1 float).
    """
    # Reshape to (1, 384) — FAISS expects a 2D array even for single queries
    q = np.array([query_vec], dtype="float32")

    distances, indices = index.search(q, min(top_k, len(rows)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            # FAISS returns -1 for padding when fewer results exist than top_k
            continue
        row = dict(rows[idx])
        row["similarity"] = float(dist)   # cosine similarity, 0–1
        results.append(row)

    return results
