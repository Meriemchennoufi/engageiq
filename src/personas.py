"""
Seed the four project personas into the database.
Each persona now has structured fields: background, interests, goal, time_budget.
The ranking vector is built from interests + goal combined.
"""
from database import upsert_persona, init_db

PERSONAS = {
    "Sofia": {
        "background":   "PhD student in NLP at UC Davis",
        "interests":    "machine learning NLP data pipelines Python pandas neural networks",
        "goal":         "Find open source repos to contribute to and build my portfolio",
        "time_budget":  10,
    },
    "David": {
        "background":   "Senior DevOps engineer with 6 years of cloud infrastructure experience",
        "interests":    "Kubernetes Terraform CI/CD observability cloud-native Helm Prometheus GitOps",
        "goal":         "Stay current with cloud-native tools and discover projects to contribute to",
        "time_budget":  8,
    },
    "Lina": {
        "background":   "Tech journalist covering emerging software trends",
        "interests":    "trending open source viral GitHub stars fast-growing communities emerging tools",
        "goal":         "Spot the next big thing before it goes mainstream",
        "time_budget":  5,
    },
    "Raj": {
        "background":   "Indie developer building a B2B SaaS product",
        "interests":    "developer tools API CLI productivity SaaS startup open source business models",
        "goal":         "Find tools and communities to grow my product and get early users",
        "time_budget":  6,
    },
}


def persona_vector_text(persona: dict) -> str:
    """Combine interests + goal into one string for embedding."""
    return f"{persona['interests']} {persona['goal']}"


def seed_personas():
    init_db()
    for name, p in PERSONAS.items():
        # Store combined interests+goal as the interests field in DB
        upsert_persona(name, persona_vector_text(p))
    print(f"Seeded {len(PERSONAS)} personas: {', '.join(PERSONAS.keys())}")


if __name__ == "__main__":
    seed_personas()
