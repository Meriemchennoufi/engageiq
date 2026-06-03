# EngageIQ — Smart Engagement Opportunity Scorer

## Setup

```bash
cd engageiq
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
```

## Run

```bash
streamlit run src/app.py
```

## First-time workflow

1. Open the dashboard in your browser
2. Click **Run Ingestion** (sidebar) — collects 10,000+ records
3. Click **Compute Embeddings** — encodes all records
4. Click **Load / Refresh Rankings** — builds FAISS index and ranks opportunities
5. Select a persona and explore the results

## Project structure

```
engageiq/
├── src/
│   ├── app.py          # Streamlit dashboard (main entry point)
│   ├── scraper.py      # GitHub, HN, Reddit ingestion
│   ├── database.py     # SQLite helpers
│   ├── embeddings.py   # sentence-transformers + FAISS
│   ├── ranking.py      # Composite scoring + UCB bandit
│   ├── analytics.py    # Batch trend analytics
│   └── personas.py     # Persona definitions
├── data/               # SQLite database (auto-created)
├── requirements.txt
└── .env                # API keys (not committed)
```
