import sqlite3
import json
import hashlib
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "engageiq.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


SEED_PATH = Path(__file__).parent.parent / "data" / "seed_opportunities.json.gz"


def _load_seed():
    """Populate DB from seed file — adds any missing records (INSERT OR IGNORE)."""
    if not SEED_PATH.exists():
        return
    import gzip, json as _json
    with gzip.open(SEED_PATH, "rt", encoding="utf-8") as f:
        records = _json.load(f)
    conn = get_conn()
    c = conn.cursor()
    c.executemany(
        """INSERT OR IGNORE INTO opportunities
           (id, title, url, body, source, domain, stars, comments, fetched_at, embedding)
           VALUES (:id,:title,:url,:body,:source,:domain,:stars,:comments,:fetched_at,:embedding)""",
        [{**r, "embedding": r.get("embedding")} for r in records]
    )
    conn.commit()
    conn.close()


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = get_conn()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS opportunities (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            url         TEXT NOT NULL,
            body        TEXT,
            source      TEXT NOT NULL,
            domain      TEXT,
            score       REAL DEFAULT 0.0,
            stars       INTEGER DEFAULT 0,
            comments    INTEGER DEFAULT 0,
            created_at  TEXT,
            embedding   TEXT,
            fetched_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            opportunity_id  TEXT NOT NULL,
            action          TEXT NOT NULL,
            persona         TEXT DEFAULT 'default',
            ts              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (opportunity_id) REFERENCES opportunities(id)
        );

        CREATE TABLE IF NOT EXISTS personas (
            name        TEXT PRIMARY KEY,
            interests   TEXT NOT NULL,
            embedding   TEXT
        );
    """)
    conn.commit()
    conn.close()


def reset_opportunities():
    """Drop all opportunities and reload from the clean seed file.
    Only called manually via the Reset button — never on startup."""
    conn = get_conn()
    conn.execute("DELETE FROM opportunities")
    conn.commit()
    conn.close()
    _load_seed()  # re-insert the clean 10k records


def url_to_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def insert_opportunity(row: dict) -> bool:
    """Returns True if inserted, False if duplicate."""
    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute("""
            INSERT OR IGNORE INTO opportunities
                (id, title, url, body, source, domain, stars, comments, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["id"], row["title"], row["url"], row.get("body", ""),
            row["source"], row.get("domain", ""), row.get("stars", 0),
            row.get("comments", 0), row.get("created_at", "")
        ))
        inserted = c.rowcount > 0
        conn.commit()
        return inserted
    finally:
        conn.close()


def save_embedding(opp_id: str, embedding: list):
    conn = get_conn()
    conn.execute(
        "UPDATE opportunities SET embedding=? WHERE id=?",
        (json.dumps(embedding), opp_id)
    )
    conn.commit()
    conn.close()


def save_persona_embedding(name: str, embedding: list):
    conn = get_conn()
    conn.execute(
        "UPDATE personas SET embedding=? WHERE name=?",
        (json.dumps(embedding), name)
    )
    conn.commit()
    conn.close()


def get_all_opportunities():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM opportunities").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_opportunities_with_embeddings():
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM opportunities WHERE embedding IS NOT NULL"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_feedback(opportunity_id: str, action: str, persona: str = "default"):
    conn = get_conn()
    conn.execute(
        "DELETE FROM feedback WHERE opportunity_id = ? AND persona = ?",
        (opportunity_id, persona)
    )
    conn.execute(
        "INSERT INTO feedback (opportunity_id, action, persona) VALUES (?, ?, ?)",
        (opportunity_id, action, persona)
    )
    conn.commit()
    conn.close()


def get_feedback(persona: str = "default"):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM feedback WHERE persona=? ORDER BY ts DESC",
        (persona,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_saved_opportunities(persona: str = "default"):
    """Return opportunities bookmarked by a persona, most recent first."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT o.id, o.title, o.url, o.source, o.domain, o.stars, o.comments, f.ts
        FROM feedback f
        JOIN opportunities o ON f.opportunity_id = o.id
        WHERE f.persona = ? AND f.action = 'bookmark'
        ORDER BY f.ts DESC
        """,
        (persona,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def upsert_persona(name: str, interests: str):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO personas (name, interests) VALUES (?, ?)",
        (name, interests)
    )
    conn.commit()
    conn.close()


def get_persona(name: str):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM personas WHERE name=?", (name,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_record_count():
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0]
    conn.close()
    return count
