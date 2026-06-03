"""
Seed the four project personas into the database.
"""
from database import upsert_persona, init_db

PERSONAS = {
    "Sofia": (
        "machine learning NLP data pipelines Python pandas beginner-friendly GitHub "
        "good first issue open source portfolio contribution neural networks"
    ),
    "David": (
        "Kubernetes Terraform CI/CD observability cloud-native infrastructure DevOps "
        "Helm Prometheus Grafana GitOps platform engineering"
    ),
    "Lina": (
        "trending open source viral GitHub stars fast-growing communities emerging tools "
        "Hacker News Reddit top posts breaking tech news velocity recency"
    ),
    "Raj": (
        "developer tools API CLI productivity SaaS startup developer productivity "
        "open source business models Reddit programming side projects"
    ),
}


def seed_personas():
    init_db()
    for name, interests in PERSONAS.items():
        upsert_persona(name, interests)
    print(f"Seeded {len(PERSONAS)} personas: {', '.join(PERSONAS.keys())}")


if __name__ == "__main__":
    seed_personas()
