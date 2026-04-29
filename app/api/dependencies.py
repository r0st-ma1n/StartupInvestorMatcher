from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services import (
    CatalogService,
    EmbeddingService,
    InvestorIndexService,
    MatchingService,
    RetrievalService,
    ScoringService,
)


@lru_cache
def get_catalog_service() -> CatalogService:
    """Build the default catalog service dependency for sample data access."""

    settings = get_settings()
    return CatalogService(
        startups_path=settings.data.startups_path,
        investors_path=settings.data.investors_path,
    )


@lru_cache
def get_matching_service() -> MatchingService:
    """Build the default matching service dependency for the API layer."""

    settings = get_settings()
    embedding_service = EmbeddingService(settings.embeddings)
    investor_index_service = InvestorIndexService(embedding_service)
    investor_index = investor_index_service.load_index(settings.data.investor_index_path)
    retrieval_service = RetrievalService(settings.retrieval)
    scoring_service = ScoringService(settings.scoring)
    return MatchingService(
        embedding_service=embedding_service,
        retrieval_service=retrieval_service,
        retrieval_settings=settings.retrieval,
        scoring_service=scoring_service,
        investor_index_service=investor_index_service,
        investor_index=investor_index,
    )
