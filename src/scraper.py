"""
Multi-source scraper: GitHub REST API, Hacker News API, Reddit (PRAW).
Runs a blocking collection loop; call run_ingestion() to populate the DB.
"""
import os
import time
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from tqdm import tqdm

from database import get_conn, init_db, insert_opportunity, url_to_id

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "EngageIQ/1.0")

DOMAINS = {
    "Machine Learning":         {"gh": ["machine-learning", "deep-learning", "neural-network"], "reddit": ["MachineLearning", "learnmachinelearning"]},
    "DevOps/K8s":               {"gh": ["kubernetes", "terraform", "devops"], "reddit": ["devops", "kubernetes"]},
    "Trending Open-Source":     {"gh": ["open-source", "awesome", "framework"], "reddit": ["opensource"]},
    "Developer Tools":          {"gh": ["developer-tools", "cli", "productivity"], "reddit": ["programming", "devtools"]},
    "Cybersecurity":            {"gh": ["security", "penetration-testing", "ctf"], "reddit": ["netsec", "cybersecurity"]},
    "Frontend (React/Web)":     {"gh": ["react", "nextjs", "typescript"], "reddit": ["reactjs", "webdev"]},
    "B2B SaaS":                 {"gh": ["saas", "api", "subscription"], "reddit": ["SaaS", "startups"]},
    "Blockchain":               {"gh": ["blockchain", "web3", "solidity"], "reddit": ["ethereum", "CryptoTechnology"]},
    "Python Data Eng":          {"gh": ["data-engineering", "apache-spark", "airflow"], "reddit": ["dataengineering"]},
    "GameDev (C++)":            {"gh": ["game-engine", "opengl", "gamedev"], "reddit": ["gamedev"]},
    "AI Research":              {"gh": ["llm", "transformer", "diffusion-model"], "reddit": ["artificial"]},
    "Embedded Systems (C/RTOS)":{"gh": ["embedded", "rtos", "microcontroller"], "reddit": ["embedded"]},
    "Cloud APIs":               {"gh": ["aws", "gcp", "azure"], "reddit": ["aws", "googlecloud"]},
    "Mobile Dev (iOS/Flutter)": {"gh": ["flutter", "ios", "swift"], "reddit": ["FlutterDev", "iOSProgramming"]},
    "Beginner Coding":          {"gh": ["beginner-friendly", "good-first-issue", "hacktoberfest"], "reddit": ["learnprogramming", "learnpython"]},
}


# ── GitHub ──────────────────────────────────────────────────────────────────

def _gh_headers():
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def scrape_github(domain: str, topics: list, per_topic: int = 50) -> int:
    inserted = 0
    for topic in topics:
        url = "https://api.github.com/search/repositories"
        params = {"q": f"topic:{topic}", "sort": "stars", "per_page": min(per_topic, 100)}
        try:
            resp = requests.get(url, headers=_gh_headers(), params=params, timeout=15)
            if resp.status_code == 403:
                print(f"  [GitHub] Rate limited — sleeping 60s")
                time.sleep(60)
                continue
            resp.raise_for_status()
            items = resp.json().get("items", [])
        except Exception as e:
            print(f"  [GitHub] Error for topic {topic}: {e}")
            continue

        for item in items:
            row = {
                "id":         url_to_id(item["html_url"]),
                "title":      item["full_name"],
                "url":        item["html_url"],
                "body":       (item.get("description") or "")[:500],
                "source":     "github",
                "domain":     domain,
                "stars":      item.get("stargazers_count", 0),
                "comments":   item.get("open_issues_count", 0),
                "created_at": item.get("created_at", ""),
            }
            if insert_opportunity(row):
                inserted += 1
        time.sleep(1)
    return inserted


def scrape_github_issues(domain: str, topics: list, per_topic: int = 30) -> int:
    """Scrape open issues tagged good-first-issue for beginner-friendly signal."""
    inserted = 0
    for topic in topics:
        url = "https://api.github.com/search/issues"
        params = {
            "q": f"label:\"good first issue\" topic:{topic} state:open",
            "sort": "created",
            "per_page": min(per_topic, 100),
        }
        try:
            resp = requests.get(url, headers=_gh_headers(), params=params, timeout=15)
            if resp.status_code == 403:
                time.sleep(60)
                continue
            resp.raise_for_status()
            items = resp.json().get("items", [])
        except Exception as e:
            print(f"  [GitHub Issues] Error: {e}")
            continue

        for item in items:
            row = {
                "id":         url_to_id(item["html_url"]),
                "title":      item.get("title", "")[:200],
                "url":        item["html_url"],
                "body":       (item.get("body") or "")[:500],
                "source":     "github_issue",
                "domain":     domain,
                "stars":      0,
                "comments":   item.get("comments", 0),
                "created_at": item.get("created_at", ""),
            }
            if insert_opportunity(row):
                inserted += 1
        time.sleep(1)
    return inserted


# ── Hacker News ─────────────────────────────────────────────────────────────

def scrape_hn(domain: str, keywords: list, max_items: int = 40) -> int:
    inserted = 0
    base = "https://hn.algolia.com/api/v1/search"
    for kw in keywords:
        try:
            resp = requests.get(base, params={"query": kw, "tags": "story", "hitsPerPage": max_items}, timeout=15)
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
        except Exception as e:
            print(f"  [HN] Error for keyword {kw}: {e}")
            continue

        for hit in hits:
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            row = {
                "id":         url_to_id(url),
                "title":      (hit.get("title") or "")[:200],
                "url":        url,
                "body":       (hit.get("story_text") or "")[:500],
                "source":     "hackernews",
                "domain":     domain,
                "stars":      hit.get("points", 0),
                "comments":   hit.get("num_comments", 0),
                "created_at": hit.get("created_at", ""),
            }
            if insert_opportunity(row):
                inserted += 1
        time.sleep(0.5)
    return inserted


# ── Reddit via Pullpush (no credentials required) ─────────────────────────────

def scrape_reddit(domain: str, subreddits: list, limit: int = 50) -> int:
    """
    Uses api.pullpush.io — a Reddit archive mirror that requires no auth.
    Falls back silently if the service is unreachable.
    """
    inserted = 0
    base = "https://api.pullpush.io/reddit/search/submission"

    for sub in subreddits:
        try:
            resp = requests.get(
                base,
                params={"subreddit": sub, "size": limit, "sort": "score"},
                timeout=15
            )
            if resp.status_code == 429:
                print(f"  [Reddit] Rate limited on r/{sub}, sleeping 30s")
                time.sleep(30)
                continue
            if resp.status_code != 200:
                print(f"  [Reddit] r/{sub} returned {resp.status_code}")
                continue
            posts = resp.json().get("data", [])
        except Exception as e:
            print(f"  [Reddit] Error for r/{sub}: {e}")
            continue

        for d in posts:
            permalink = d.get("permalink", "")
            post_url = (
                d["url"] if d.get("url", "").startswith("http") and not d.get("is_self")
                else f"https://reddit.com{permalink}"
            )
            row = {
                "id":         url_to_id(post_url),
                "title":      (d.get("title") or "")[:200],
                "url":        post_url,
                "body":       (d.get("selftext") or "")[:500],
                "source":     "reddit",
                "domain":     domain,
                "stars":      d.get("score", 0),
                "comments":   d.get("num_comments", 0),
                "created_at": datetime.fromtimestamp(
                    d.get("created_utc", 0), tz=timezone.utc
                ).isoformat(),
            }
            if insert_opportunity(row):
                inserted += 1
        time.sleep(1)
    return inserted


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_ingestion(target: int = 10000):
    init_db()
    from database import get_record_count
    print(f"\n=== EngageIQ Ingestion | target: {target:,} records ===\n")

    for domain, sources in tqdm(DOMAINS.items(), desc="Domains"):
        count_before = get_record_count()
        print(f"\n[{domain}]")

        gh_inserted  = scrape_github(domain, sources["gh"], per_topic=80)
        gh_inserted += scrape_github_issues(domain, sources["gh"][:2], per_topic=40)
        hn_inserted  = scrape_hn(domain, sources["gh"][:3], max_items=50)
        rd_inserted  = scrape_reddit(domain, sources["reddit"], limit=100)

        total = gh_inserted + hn_inserted + rd_inserted
        print(f"  → GH:{gh_inserted}  HN:{hn_inserted}  Reddit:{rd_inserted}  | running total: {get_record_count():,}")

        if get_record_count() >= target:
            print(f"\n Target of {target:,} reached.")
            break

    final = get_record_count()
    print(f"\n=== Ingestion complete: {final:,} records ===")
    return final


if __name__ == "__main__":
    run_ingestion()
