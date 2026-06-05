"""
Batch analytics: trend detection, topic distributions, engagement volume.
All computed from the SQLite database using pandas.
"""
import pandas as pd
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "engageiq.db"


def load_df() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM opportunities", conn, parse_dates=["fetched_at"])
    conn.close()
    return df


def top_domains(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    return (
        df.groupby("domain")
        .agg(count=("id", "count"), avg_stars=("stars", "mean"), total_comments=("comments", "sum"))
        .reset_index()
        .sort_values("count", ascending=False)
        .head(n)
    )


def source_distribution(df: pd.DataFrame) -> pd.DataFrame:
    # Merge github_issue into github for cleaner display
    df2 = df.copy()
    df2["source"] = df2["source"].replace("github_issue", "github")
    vc = df2["source"].value_counts()
    return pd.DataFrame({"source": vc.index, "count": vc.values})


def top_opportunities(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return df.nlargest(n, "stars")[["title", "url", "source", "domain", "stars", "comments"]]


def trending_by_stars(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top repos by stars as a proxy for trend velocity."""
    return (
        df[df["source"] == "github"]
        .nlargest(n, "stars")[["title", "domain", "stars", "comments", "url"]]
        .reset_index(drop=True)
    )


def engagement_volume_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """Records fetched per day — shows ingestion timeline."""
    df2 = df.copy()
    df2["date"] = df2["fetched_at"].dt.date
    return df2.groupby("date").size().reset_index(name="count")


def domain_star_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["domain", "source"])["stars"]
        .mean()
        .reset_index()
        .pivot(index="domain", columns="source", values="stars")
        .fillna(0)
    )


def summary_stats(df: pd.DataFrame) -> dict:
    return {
        "total_records":   len(df),
        "unique_domains":  df["domain"].nunique(),
        "sources":         df["source"].value_counts().to_dict(),
        "avg_stars":       round(df["stars"].mean(), 1),
        "avg_comments":    round(df["comments"].mean(), 1),
        "top_domain":      df["domain"].value_counts().idxmax() if len(df) else "—",
    }
