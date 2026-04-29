from __future__ import annotations

import numpy as np
import pytest

from app.config import ScoringSettings
from app.config import RetrievalSettings
from app.models import Candidate, InvestorProfile, StartupProfile
from app.services import MatchingService, RetrievalService, ScoringService


class StubEmbeddingService:
    def __init__(self, startup_embeddings: np.ndarray, investor_embeddings: np.ndarray) -> None:
        self._startup_embeddings = startup_embeddings
        self._investor_embeddings = investor_embeddings
        self.calls: list[tuple[str, int]] = []

    def embed_startups(self, startups):
        self.calls.append(("startups", len(startups)))
        return self._startup_embeddings

    def embed_investors(self, investors):
        self.calls.append(("investors", len(investors)))
        return self._investor_embeddings


def test_generate_candidates_uses_candidate_pool_size() -> None:
    startup = StartupProfile(
        startup_id="s1",
        name="Acme AI",
        description="AI tooling",
    )
    investors = [
        InvestorProfile(investor_id="i1", name="Alpha", description="Generalist"),
        InvestorProfile(investor_id="i2", name="Beta", description="AI fund"),
        InvestorProfile(investor_id="i3", name="Gamma", description="Climate fund"),
    ]
    embedding_service = StubEmbeddingService(
        startup_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        investor_embeddings=np.array(
            [
                [0.6, 0.8],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    service = MatchingService(
        embedding_service=embedding_service,
        retrieval_service=RetrievalService(RetrievalSettings(default_top_k=2, candidate_pool_size=3)),
        retrieval_settings=RetrievalSettings(default_top_k=2, candidate_pool_size=2),
    )

    candidates = service.generate_candidates(startup, investors)

    assert len(candidates) == 2
    assert candidates[0].investor.investor_id == "i2"
    assert candidates[1].investor.investor_id == "i1"
    assert embedding_service.calls == [("startups", 1), ("investors", 3)]


def test_match_startup_semantic_uses_top_k() -> None:
    startup = StartupProfile(
        startup_id="s1",
        name="Acme AI",
        description="AI tooling",
    )
    investors = [
        InvestorProfile(investor_id="i1", name="Alpha", description="Generalist"),
        InvestorProfile(investor_id="i2", name="Beta", description="AI fund"),
        InvestorProfile(investor_id="i3", name="Gamma", description="Climate fund"),
    ]
    embedding_service = StubEmbeddingService(
        startup_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        investor_embeddings=np.array(
            [
                [0.6, 0.8],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    service = MatchingService(
        embedding_service=embedding_service,
        retrieval_service=RetrievalService(RetrievalSettings(default_top_k=3, candidate_pool_size=3)),
        retrieval_settings=RetrievalSettings(default_top_k=3, candidate_pool_size=3),
    )

    matches = service.match_startup_semantic(startup, investors, top_k=1)

    assert len(matches) == 1
    assert matches[0].investor.investor_id == "i2"


def test_matching_service_returns_empty_for_no_investors() -> None:
    startup = StartupProfile(
        startup_id="s1",
        name="Acme AI",
        description="AI tooling",
    )
    embedding_service = StubEmbeddingService(
        startup_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        investor_embeddings=np.empty((0, 2), dtype=np.float32),
    )
    service = MatchingService(
        embedding_service=embedding_service,
        retrieval_service=RetrievalService(RetrievalSettings()),
        retrieval_settings=RetrievalSettings(),
    )

    candidates = service.generate_candidates(startup, [])
    matches = service.match_startup_semantic(startup, [], top_k=5)

    assert candidates == []
    assert matches == []
    assert embedding_service.calls == []


def test_rerank_candidates_orders_by_weighted_score() -> None:
    startup = StartupProfile(
        startup_id="s1",
        name="Acme AI",
        description="AI tooling",
        industries=["AI"],
        stage="Seed",
        country="US",
        region="North America",
        fundraising_amount=1_000_000,
    )
    candidates = [
        Candidate(
            investor=InvestorProfile(
                investor_id="i1",
                name="Alpha",
                description="Generalist",
                industries=["AI"],
                preferred_stages=["Seed"],
                countries=["US"],
                ticket_min=200_000,
                ticket_max=2_000_000,
            ),
            semantic_similarity=0.7,
            retrieval_rank=2,
        ),
        Candidate(
            investor=InvestorProfile(
                investor_id="i2",
                name="Beta",
                description="Generalist",
                industries=["Climate"],
                preferred_stages=["Series A"],
                countries=["UK"],
                ticket_min=5_000_000,
                ticket_max=10_000_000,
            ),
            semantic_similarity=0.95,
            retrieval_rank=1,
        ),
    ]
    service = MatchingService(
        embedding_service=StubEmbeddingService(
            startup_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
            investor_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        ),
        retrieval_service=RetrievalService(RetrievalSettings()),
        retrieval_settings=RetrievalSettings(),
        scoring_service=ScoringService(ScoringSettings()),
    )

    results = service.rerank_candidates(startup, candidates)

    assert len(results) == 2
    assert results[0].investor_id == "i1"
    assert results[0].candidate.retrieval_rank == 2
    assert results[0].rank == 1
    assert results[1].investor_id == "i2"


def test_match_startup_runs_end_to_end_semantic_retrieval_and_reranking() -> None:
    startup = StartupProfile(
        startup_id="s1",
        name="Acme AI",
        description="AI tooling",
        industries=["AI"],
        stage="Seed",
        country="US",
        region="North America",
        fundraising_amount=1_000_000,
    )
    investors = [
        InvestorProfile(
            investor_id="i1",
            name="Alpha",
            description="Generalist",
            industries=["Climate"],
            preferred_stages=["Series A"],
            countries=["UK"],
            ticket_min=5_000_000,
            ticket_max=10_000_000,
        ),
        InvestorProfile(
            investor_id="i2",
            name="Beta",
            description="AI specialist",
            industries=["AI"],
            preferred_stages=["Seed"],
            countries=["US"],
            ticket_min=200_000,
            ticket_max=2_000_000,
        ),
    ]
    service = MatchingService(
        embedding_service=StubEmbeddingService(
            startup_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
            investor_embeddings=np.array(
                [
                    [1.0, 0.0],
                    [0.8, 0.6],
                ],
                dtype=np.float32,
            ),
        ),
        retrieval_service=RetrievalService(RetrievalSettings(default_top_k=2, candidate_pool_size=2)),
        retrieval_settings=RetrievalSettings(default_top_k=2, candidate_pool_size=2),
        scoring_service=ScoringService(ScoringSettings()),
    )

    results = service.match_startup(startup, investors, top_k=1, candidate_pool_size=2)

    assert len(results) == 1
    assert results[0].investor_id == "i2"
    assert results[0].score_breakdown.stage_match == 1.0


def test_rerank_candidates_rejects_invalid_top_k() -> None:
    startup = StartupProfile(
        startup_id="s1",
        name="Acme AI",
        description="AI tooling",
    )
    candidate = Candidate(
        investor=InvestorProfile(
            investor_id="i1",
            name="Alpha",
            description="Generalist",
        ),
        semantic_similarity=0.8,
        retrieval_rank=1,
    )
    service = MatchingService(
        embedding_service=StubEmbeddingService(
            startup_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
            investor_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        ),
        retrieval_service=RetrievalService(RetrievalSettings()),
        retrieval_settings=RetrievalSettings(),
        scoring_service=ScoringService(ScoringSettings()),
    )

    with pytest.raises(ValueError):
        service.rerank_candidates(startup, [candidate], top_k=0)
