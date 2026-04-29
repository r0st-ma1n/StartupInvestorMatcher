from __future__ import annotations

from collections.abc import Sequence

from app.config import RetrievalSettings
from app.models import Candidate, InvestorProfile, MatchResult, StartupProfile
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService
from app.services.scoring_service import ScoringService


class MatchingService:
    """Coordinates semantic candidate generation for startup-investor matching."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        retrieval_service: RetrievalService,
        retrieval_settings: RetrievalSettings,
        scoring_service: ScoringService | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._retrieval_service = retrieval_service
        self._retrieval_settings = retrieval_settings
        self._scoring_service = scoring_service

    def generate_candidates(
        self,
        startup: StartupProfile,
        investors: Sequence[InvestorProfile],
        candidate_pool_size: int | None = None,
    ) -> list[Candidate]:
        """Generate a semantic candidate pool for later reranking."""

        if not investors:
            return []

        pool_size = (
            self._retrieval_settings.candidate_pool_size
            if candidate_pool_size is None
            else candidate_pool_size
        )
        return self._retrieve_candidates(startup=startup, investors=investors, top_k=pool_size)

    def match_startup_semantic(
        self,
        startup: StartupProfile,
        investors: Sequence[InvestorProfile],
        top_k: int | None = None,
    ) -> list[Candidate]:
        """Return top-k investors using semantic retrieval only."""

        if not investors:
            return []

        return self._retrieve_candidates(startup=startup, investors=investors, top_k=top_k)

    def rerank_candidates(
        self,
        startup: StartupProfile,
        candidates: Sequence[Candidate],
        top_k: int | None = None,
    ) -> list[MatchResult]:
        """Rerank retrieved candidates using rule-based business scoring."""

        if not candidates:
            return []
        if self._scoring_service is None:
            raise ValueError("scoring_service is required for reranking.")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be greater than 0.")

        scored_candidates = []
        for candidate in candidates:
            score_breakdown = self._scoring_service.score_candidate(startup, candidate)
            scored_candidates.append((candidate, score_breakdown))

        scored_candidates.sort(
            key=lambda item: (
                item[1].weighted_score,
                item[0].semantic_similarity,
                -item[0].retrieval_rank,
            ),
            reverse=True,
        )

        limit = len(scored_candidates) if top_k is None else top_k
        results: list[MatchResult] = []
        for rank, (candidate, score_breakdown) in enumerate(scored_candidates[:limit], start=1):
            results.append(
                MatchResult(
                    startup_id=startup.startup_id,
                    investor_id=candidate.investor.investor_id,
                    investor_name=candidate.investor.name,
                    rank=rank,
                    score=score_breakdown.weighted_score,
                    candidate=candidate,
                    score_breakdown=score_breakdown,
                )
            )
        return results

    def match_startup(
        self,
        startup: StartupProfile,
        investors: Sequence[InvestorProfile],
        top_k: int | None = None,
        candidate_pool_size: int | None = None,
    ) -> list[MatchResult]:
        """Run semantic retrieval followed by rule-based reranking."""

        candidates = self.generate_candidates(
            startup=startup,
            investors=investors,
            candidate_pool_size=candidate_pool_size,
        )
        return self.rerank_candidates(startup=startup, candidates=candidates, top_k=top_k)

    def _retrieve_candidates(
        self,
        startup: StartupProfile,
        investors: Sequence[InvestorProfile],
        top_k: int | None,
    ) -> list[Candidate]:
        startup_embedding = self._embedding_service.embed_startups([startup])[0]
        investor_embeddings = self._embedding_service.embed_investors(investors)
        return self._retrieval_service.retrieve_top_k_candidates(
            startup=startup,
            startup_embedding=startup_embedding,
            investors=investors,
            investor_embeddings=investor_embeddings,
            top_k=top_k,
        )
