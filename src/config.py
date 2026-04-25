"""Shared HTTP settings for arXiv, Crossref, and Semantic Scholar APIs."""

from __future__ import annotations

import os

# API Endpoints
ARXIV_API = "https://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

# arXiv asks for a descriptive User-Agent with contact info when possible.
_DEFAULT_UA = "refsproj1/1.0 (+https://arxiv.org/help/contact; mailto:{email})"


def request_delay(api: str = "arxiv") -> float:
    """Return request delay for the specified API in seconds.

    Args:
        api: API name ('arxiv', 'crossref', 'semantic_scholar')

    Returns:
        Delay in seconds
    """
    # Default delays by API
    defaults = {
        "arxiv": 3.0,
        "crossref": 0.5,  # Crossref is more lenient with polite pool
        "semantic_scholar": 1.0,  # S2: 100 req/5min without key
    }

    # Check for API-specific env var first
    env_var = f"{api.upper()}_REQUEST_DELAY"
    delay = os.environ.get(env_var)
    if delay:
        return float(delay)

    # Fall back to generic env var or default
    return float(os.environ.get("ARXIV_REQUEST_DELAY", defaults.get(api, 3.0)))


def default_headers() -> dict[str, str]:
    email = os.environ.get("ARXIV_CONTACT_EMAIL", "your-email@example.com")
    return {"User-Agent": _DEFAULT_UA.format(email=email)}
