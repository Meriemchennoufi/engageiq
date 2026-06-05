"""
Seed the four project personas into the database.
Each persona matches the official BAX-423 project specification exactly.
"""
from database import upsert_persona, init_db

PERSONAS = {
    "Sofia": {
        "background":   "MSBA student, graduating soon. Wants to build a visible open-source portfolio before job hunting.",
        "interests":    "machine learning NLP data pipelines Python pandas beginner-friendly good first issue open source",
        "goal":         "Find beginner-friendly GitHub repos to contribute to and Reddit/blog discussions to engage with for visibility.",
        "time_budget":  5,
    },
    "David": {
        "background":   "Mid-career DevOps engineer (5 years). Wants to establish thought leadership in cloud-native infrastructure.",
        "interests":    "Kubernetes Terraform CI/CD observability cloud-native infrastructure Helm Prometheus GitOps",
        "goal":         "Find high-signal GitHub projects, Reddit threads, and blog posts where expert commentary adds value.",
        "time_budget":  3,
    },
    "Lina": {
        "background":   "Data journalist at a tech publication. Monitors open-source and tech communities for story leads.",
        "interests":    "trending repos viral discussions emerging tools community fast-growing stars recency velocity",
        "goal":         "Surface fast-growing repos, trending Reddit threads, and blog posts gaining traction before they go mainstream.",
        "time_budget":  10,
    },
    "Raj": {
        "background":   "Technical co-founder of a developer tools startup. Wants to grow awareness by engaging in relevant communities.",
        "interests":    "developer productivity APIs CLI tools open-source business models SaaS startup side projects",
        "goal":         "Find Reddit threads and blog posts where his product is relevant, and GitHub repos where integration makes sense.",
        "time_budget":  4,
    },
}


def persona_vector_text(persona: dict) -> str:
    """Combine interests + goal into one string for embedding."""
    return f"{persona['interests']} {persona['goal']}"


def seed_personas():
    init_db()
    for name, p in PERSONAS.items():
        upsert_persona(name, persona_vector_text(p))
    print(f"Seeded {len(PERSONAS)} personas: {', '.join(PERSONAS.keys())}")


if __name__ == "__main__":
    seed_personas()
