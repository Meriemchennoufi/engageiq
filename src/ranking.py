"""
ranking.py — Multi-stage pipeline that turns raw similarity scores into a
             personalised, adaptive ranked list of engagement opportunities.

THE THREE STAGES:
─────────────────
  Stage 1 — RETRIEVAL (embeddings.py / FAISS)
    Fast approximate search to pull the most semantically similar records
    for a persona from the full database. Produces a "candidate set".

  Stage 2 — COMPOSITE SCORING (this file)
    Each candidate gets a score combining three signals:
      • Relevance  (50%) — cosine similarity between persona and opportunity
      • Community  (30%) — log-scaled stars + comments (popularity signal)
      • Recency    (20%) — exponential decay; newer = higher score

  Stage 3 — UCB BANDIT RE-RANKING (this file)
    A reinforcement-learning layer that adjusts scores based on the user's
    past Engage / Skip / Save actions. Items the user engaged with get a
    boost; skipped items are penalised. Unexplored items get an "optimism
    bonus" so the system keeps surfacing new opportunities.
"""

import json
import math
import numpy as np
from datetime import datetime, timezone

from database import get_feedback


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — COMPOSITE SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _recency_score(created_at: str) -> float:
    """
    Return a recency score between 0 and 1 using exponential decay.

    Formula: score = e^(-age_in_days / 90)

    Why exponential decay?
      - An item created today scores 1.0 (e^0 = 1).
      - After 90 days it scores ~0.37 (e^-1 ≈ 0.37).
      - After 6 months it scores ~0.14.
      - Very old items approach 0 but never reach it.

    The half-life of 90 days is a design choice — it favours recent
    activity without completely burying older high-quality content.

    If created_at is missing we return a neutral 0.3 (assume moderately old).
    """
    if not created_at:
        return 0.3
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - dt).days
        return math.exp(-age_days / 90)
    except Exception:
        return 0.3


def _community_score(stars: int, comments: int) -> float:
    """
    Return a community health score between 0 and 1.

    WHY LOGARITHM?
      Raw star counts span many orders of magnitude — a repo with 200k stars
      vs. one with 200 stars. If we used raw counts directly, the top-starred
      repos would completely dominate. log1p(x) = log(1 + x) compresses this
      range so that:
        - 0 stars    → log1p(0)    = 0.0
        - 100 stars  → log1p(100)  ≈ 4.6
        - 10,000     → log1p(10k)  ≈ 9.2
        - 100,000    → log1p(100k) ≈ 11.5

      Dividing by 15 (for stars) and 10 (for comments) scales to roughly 0–1.
      min(1.0, ...) caps it so outliers (tensorflow: 185k stars) don't exceed 1.
    """
    s = math.log1p(stars)   / 15.0
    c = math.log1p(comments) / 10.0
    return min(1.0, (s + c) / 2)


def composite_score(row: dict, similarity: float) -> float:
    """
    Combine relevance, community, and recency into a single score.

    WEIGHTS (tunable design decisions):
      50% relevance  — the most important signal: does this match what you care about?
      30% community  — high stars/comments means quality content and active discussion
      20% recency    — newer = more timely engagement opportunity

    These weights could be made user-configurable in a future version.
    """
    relevance = similarity                               # 0–1 cosine similarity from embeddings
    community = _community_score(row.get("stars", 0), row.get("comments", 0))
    recency   = _recency_score(row.get("created_at", ""))

    score = (0.50 * relevance) + (0.30 * community) + (0.20 * recency)
    return round(score, 4)


def explain(row: dict, similarity: float) -> str:
    """
    Build a human-readable explanation of WHY this opportunity was recommended.
    Shown in the "Why:" block on each card in the UI.

    This is intentionally short — just the 2–4 most notable facts.
    """
    parts = []

    # Always show relevance percentage
    parts.append(f"Relevance: {similarity:.0%} match to your interests")

    # Only mention stars if noteworthy (> 100)
    if row.get("stars", 0) > 100:
        parts.append(f"{row['stars']:,} stars — active community")

    # Only mention comments if there's real discussion (> 10)
    if row.get("comments", 0) > 10:
        parts.append(f"{row['comments']} open discussions")

    # Source-specific context
    if row.get("source") == "github_issue":
        parts.append("Open issue — good entry point to contribute")

    # Flag very fresh content (recency > 0.8 ≈ posted within ~16 days)
    recency = _recency_score(row.get("created_at", ""))
    if recency > 0.8:
        parts.append("Very recent — high visibility window")

    return " · ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — UCB BANDIT RE-RANKING
# ══════════════════════════════════════════════════════════════════════════════

class UCBBandit:
    """
    Upper Confidence Bound (UCB1) bandit that personalises rankings using
    feedback the user has given (Engage / Skip / Save for later).

    WHAT IS A BANDIT ALGORITHM?
    ────────────────────────────
    A "multi-armed bandit" is a reinforcement learning framework for
    the exploration vs. exploitation trade-off:

      • Exploitation — show items similar to what the user liked before
                       (maximise known rewards)
      • Exploration  — occasionally surface new/untried items in case they
                       turn out to be great (discover unknown rewards)

    UCB1 handles this mathematically by adding an "uncertainty bonus"
    to each item's score. Items we've shown many times have low uncertainty
    (the bonus shrinks). Items never shown have high uncertainty (big bonus),
    so the algorithm keeps trying new things.

    REWARD MAPPING:
      engage   → 1.0  (strong positive signal — user found it worth clicking)
      bookmark → 0.7  (moderate signal — interesting enough to save)
      skip     → 0.0  (negative signal — not relevant to user)
    """

    def __init__(self, alpha: float = 1.0):
        # alpha controls exploration aggressiveness.
        # Higher alpha = more willing to try unseen items.
        # alpha=1.0 is the standard UCB1 setting.
        self.alpha = alpha
        self.counts:  dict[str, int]   = {}  # how many times each opportunity was actioned
        self.rewards: dict[str, float] = {}  # cumulative reward per opportunity

    def load_feedback(self, persona: str = "default"):
        """
        Read all past feedback from the database and populate counts/rewards.
        Called fresh each time we rank, so new feedback is always reflected.
        """
        feedback = get_feedback(persona)
        for fb in feedback:
            oid    = fb["opportunity_id"]
            action = fb["action"]
            # Map each action to a numeric reward
            reward = {"engage": 1.0, "bookmark": 0.7, "skip": 0.0}.get(action, 0.0)
            self.counts[oid]  = self.counts.get(oid, 0) + 1
            self.rewards[oid] = self.rewards.get(oid, 0.0) + reward

    def ucb_score(self, opp_id: str, total_rounds: int, base_score: float) -> float:
        """
        Compute the UCB-adjusted final score for one opportunity.

        FORMULA:
          final = base_score + 0.3 × mean_reward + 0.2 × exploration_bonus

        Where:
          mean_reward       = total_rewards / times_shown
                              (how good was this item on average?)
          exploration_bonus = alpha × sqrt(log(total_rounds + 1) / n)
                              (how uncertain are we? — shrinks as n grows)

        For unseen items (n=0):
          We return base_score + alpha (maximum optimism — try everything once).
          This is the "optimistic initialisation" strategy in UCB.

        WHY ADD TO base_score INSTEAD OF REPLACING IT?
          The composite score already encodes relevance + community + recency.
          The bandit *adjusts* it based on learned preferences, rather than
          overriding all that signal.
        """
        n = self.counts.get(opp_id, 0)

        # Unseen item — be maximally optimistic to encourage exploration
        if n == 0:
            return base_score + self.alpha

        mean_reward       = self.rewards[opp_id] / n
        exploration_bonus = self.alpha * math.sqrt(math.log(total_rounds + 1) / n)

        return base_score + 0.3 * mean_reward + 0.2 * exploration_bonus

    def rank(self, candidates: list, persona: str = "default") -> list:
        """
        Apply UCB scoring to all candidates and return them sorted best-first.

        Steps:
          1. Load feedback from DB for this persona.
          2. Compute total_rounds = total actions taken (denominator for UCB formula).
          3. For each candidate, compute its final_score via ucb_score().
          4. Sort descending by final_score.
        """
        self.load_feedback(persona)
        total_rounds = max(1, sum(self.counts.values()))  # avoid log(0)

        for c in candidates:
            c["final_score"] = self.ucb_score(
                c["id"],
                total_rounds,
                c.get("composite_score", 0.0)
            )

        return sorted(candidates, key=lambda x: x["final_score"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE — used when loading all records via FAISS
# ══════════════════════════════════════════════════════════════════════════════

def rank_opportunities(
    persona_vec: list,
    index,
    rows: list,
    persona_name: str = "default",
    top_n: int = 20,
) -> list:
    """
    Run the complete 3-stage ranking pipeline using a pre-built FAISS index.

    Used at "Update Rankings" time when we want to build the global ranked list.

    Stage 1: FAISS retrieves top 200 candidates by cosine similarity.
             (200 gives plenty of headroom for re-ranking)
    Stage 2: Each candidate gets a composite score (relevance + community + recency).
    Stage 3: UCB bandit re-ranks using the persona's past feedback.
    Returns: top_n final results.
    """
    from embeddings import search

    # Stage 1 — FAISS retrieval (fast vector search)
    candidates = search(persona_vec, index, rows, top_k=200)

    # Stage 2 — Composite scoring + explanation generation
    for c in candidates:
        c["composite_score"] = composite_score(c, c["similarity"])
        c["explanation"]     = explain(c, c["similarity"])

    # Stage 3 — UCB bandit re-ranking
    bandit = UCBBandit()
    ranked = bandit.rank(candidates, persona=persona_name)

    return ranked[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# FILTER-FIRST PIPELINE — used when the user applies Source / Domain filters
# ══════════════════════════════════════════════════════════════════════════════

def rank_from_rows(
    persona_vec: list,
    rows: list,
    persona_name: str = "default",
    top_n: int = 15,
) -> list:
    """
    Rank a pre-filtered list of rows directly using numpy — no FAISS needed.

    WHY A DIFFERENT FUNCTION?
      When the user applies a filter (e.g. "Reddit only" or "AI Research domain"),
      we want to rank within that filtered pool — not just trim a global top 50.
      Rebuilding the FAISS index for every filter change would be slow.

      Instead, since all our embedding vectors are already L2-normalised,
      cosine similarity = dot product, which numpy can compute in microseconds
      even for thousands of rows.

    FLOW:
      1. Convert persona_vec to a numpy array.
      2. For each row, parse its stored embedding and compute dot product.
      3. Compute composite score (relevance + community + recency).
      4. UCB bandit re-ranks based on feedback.
      5. Return top_n.

    This means filters are genuinely powerful:
      "Reddit + Machine Learning" → searches ALL Reddit ML records,
      not just the few that happened to be in a global top 50.
    """
    if not rows or not persona_vec:
        return []

    pvec = np.array(persona_vec, dtype="float32")
    candidates = []

    for row in rows:
        emb = row.get("embedding")
        if not emb:
            # Skip rows without embeddings (they can't be ranked by similarity)
            continue

        # Parse embedding — stored as JSON string in SQLite, or already a list
        vec = np.array(json.loads(emb) if isinstance(emb, str) else emb, dtype="float32")

        # Cosine similarity via dot product (valid because vectors are L2-normalised)
        sim = float(np.dot(pvec, vec))

        c = dict(row)
        c["similarity"]      = sim
        c["composite_score"] = composite_score(c, sim)
        c["explanation"]     = explain(c, sim)
        candidates.append(c)

    # UCB bandit re-ranks the filtered candidates
    bandit = UCBBandit()
    ranked = bandit.rank(candidates, persona=persona_name)

    return ranked[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION METRIC — NDCG@k
# ══════════════════════════════════════════════════════════════════════════════

def ndcg_at_k(ranked: list, relevant_ids: set, k: int = 10) -> float:
    """
    Compute NDCG@k (Normalised Discounted Cumulative Gain at k).

    WHAT IS NDCG?
      A standard information retrieval metric that measures ranking quality.
      It rewards putting relevant items at the top of the list more than at the bottom.

      DCG (Discounted CG) = sum of (relevance / log2(rank + 1))
        → items at rank 1 contribute more than items at rank 5.

      NDCG = DCG / IDCG
        where IDCG is the DCG of the ideal (perfect) ranking.
        NDCG = 1.0 means perfect ranking; 0.0 means all relevant items at the bottom.

    In EngageIQ, relevant_ids = set of opportunity IDs the user has engaged with.
    We use this in the simulation to show how the bandit improves over time.
    """
    def dcg(items):
        score = 0.0
        for i, item in enumerate(items[:k]):
            rel = 1 if item["id"] in relevant_ids else 0
            score += rel / math.log2(i + 2)   # i+2 because log2(1) = 0
        return score

    # IDCG: the best possible DCG if all relevant items were shown first
    ideal = sorted(ranked, key=lambda x: x["id"] in relevant_ids, reverse=True)
    idcg  = dcg(ideal)

    return round(dcg(ranked) / idcg, 4) if idcg > 0 else 0.0
