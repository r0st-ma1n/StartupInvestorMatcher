from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services import EmbeddingService, MatchingService, RetrievalService, ScoringService


@lru_cache
def get_matching_service() -> MatchingService:
    """Build the default matching service dependency for the API layer."""

    settings = get_settings()
    embedding_service = EmbeddingService(settings.embeddings)
    retrieval_service = RetrievalService(settings.retrieval)
    scoring_service = ScoringService(settings.scoring)
    return MatchingService(
        embedding_service=embedding_service,
        retrieval_service=retrieval_service,
        retrieval_settings=settings.retrieval,
        scoring_service=scoring_service,
    )
