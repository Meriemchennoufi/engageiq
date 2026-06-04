"""
Multi-stage ranking pipeline:
  1. Candidate retrieval via FAISS (cosine similarity)
  2. Composite scoring (relevance + community health + recency)
  3. Re-ranking with UCB bandit adjusted by feedback
"""
import json
import math
import numpy as np
from datetime import datetime, timezone

from database import get_feedback


# ── Composite scorer ─────────────────────────────────────────────────────────

def _recency_score(created_at: str) -> float:
    """Decay score: 1.0 if today, approaching 0 for old items."""
    if not created_at:
        return 0.3
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - dt).days
        return math.exp(-age_days / 90)
    except Exception:
        return 0.3


def _community_score(stars: int, comments: int) -> float:
    """Log-normalized community health signal."""
    s = math.log1p(stars) / 15.0
    c = math.log1p(comments) / 10.0
    return min(1.0, (s + c) / 2)


def composite_score(row: dict, similarity: float) -> float:
    relevance  = similarity                              # 0–1 cosine sim
    community  = _community_score(row.get("stars", 0), row.get("comments", 0))
    recency    = _recency_score(row.get("created_at", ""))

    score = (0.50 * relevance) + (0.30 * community) + (0.20 * recency)
    return round(score, 4)


def explain(row: dict, similarity: float) -> str:
    parts = []
    parts.append(f"Relevance: {similarity:.0%} match to your interests")
    if row.get("stars", 0) > 100:
        parts.append(f"{row['stars']:,} stars — active community")
    if row.get("comments", 0) > 10:
        parts.append(f"{row['comments']} open discussions")
    if row.get("source") == "github_issue":
        parts.append("Open issue — good entry point to contribute")
    recency = _recency_score(row.get("created_at", ""))
    if recency > 0.8:
        parts.append("Very recent — high visibility window")
    return " · ".join(parts)


# ── UCB Bandit re-ranker ──────────────────────────────────────────────────────

class UCBBandit:
    """
    Upper Confidence Bound bandit that re-ranks candidates based on
    accumulated engage/skip feedback per opportunity.
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.counts:  dict[str, int]   = {}  # total shown
        self.rewards: dict[str, float] = {}  # cumulative reward

    def load_feedback(self, persona: str = "default"):
        feedback = get_feedback(persona)
        for fb in feedback:
            oid    = fb["opportunity_id"]
            action = fb["action"]
            reward = {"engage": 1.0, "bookmark": 0.7, "skip": 0.0}.get(action, 0.0)
            self.counts[oid]  = self.counts.get(oid, 0) + 1
            self.rewards[oid] = self.rewards.get(oid, 0.0) + reward

    def ucb_score(self, opp_id: str, total_rounds: int, base_score: float) -> float:
        n = self.counts.get(opp_id, 0)
        if n == 0:
            return base_score + self.alpha  # unexplored — optimistic
        mean   = self.rewards[opp_id] / n
        bonus  = self.alpha * math.sqrt(math.log(total_rounds + 1) / n)
        return base_score + 0.3 * mean + 0.2 * bonus

    def rank(self, candidates: list, persona: str = "default") -> list:
        self.load_feedback(persona)
        total_rounds = max(1, sum(self.counts.values()))
        for c in candidates:
            oid        = c["id"]
            base       = c.get("composite_score", 0.0)
            c["final_score"] = self.ucb_score(oid, total_rounds, base)
        return sorted(candidates, key=lambda x: x["final_score"], reverse=True)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def rank_opportunities(
    persona_vec: list,
    index,
    rows: list,
    persona_name: str = "default",
    top_n: int = 20,
) -> list:
    from embeddings import search

    candidates = search(persona_vec, index, rows, top_k=200)

    for c in candidates:
        c["composite_score"] = composite_score(c, c["similarity"])
        c["explanation"]     = explain(c, c["similarity"])

    bandit = UCBBandit()
    ranked = bandit.rank(candidates, persona=persona_name)
    return ranked[:top_n]


# ── Filter-first ranking (no FAISS rebuild needed) ───────────────────────────

def rank_from_rows(
    persona_vec: list,
    rows: list,
    persona_name: str = "default",
    top_n: int = 15,
) -> list:
    """
    Rank a pre-filtered list of rows directly using numpy dot product.
    Vectors must already be L2-normalised (they are, from sentence-transformers).
    """
    if not rows or not persona_vec:
        return []

    pvec = np.array(persona_vec, dtype="float32")
    candidates = []
    for row in rows:
        emb = row.get("embedding")
        if not emb:
            continue
        vec = np.array(json.loads(emb) if isinstance(emb, str) else emb, dtype="float32")
        sim = float(np.dot(pvec, vec))
        c = dict(row)
        c["similarity"]      = sim
        c["composite_score"] = composite_score(c, sim)
        c["explanation"]     = explain(c, sim)
        candidates.append(c)

    bandit = UCBBandit()
    ranked = bandit.rank(candidates, persona=persona_name)
    return ranked[:top_n]


# ── Evaluation metric (NDCG@10) ──────────────────────────────────────────────

def ndcg_at_k(ranked: list, relevant_ids: set, k: int = 10) -> float:
    """Compute NDCG@k given a set of relevant opportunity IDs."""
    def dcg(items):
        score = 0.0
        for i, item in enumerate(items[:k]):
            rel = 1 if item["id"] in relevant_ids else 0
            score += rel / math.log2(i + 2)
        return score

    ideal = sorted(ranked, key=lambda x: x["id"] in relevant_ids, reverse=True)
    idcg  = dcg(ideal)
    return round(dcg(ranked) / idcg, 4) if idcg > 0 else 0.0
