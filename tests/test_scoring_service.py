from __future__ import annotations

import pytest

from app.config import ScoringSettings
from app.models import Candidate, InvestorProfile, StartupProfile
from app.services import ScoringService


def test_score_candidate_aggregates_weighted_components() -> None:
    service = ScoringService(
        ScoringSettings(
            semantic_similarity_weight=0.5,
            industry_match_weight=0.2,
            stage_match_weight=0.1,
            geo_match_weight=0.1,
            ticket_size_fit_weight=0.1,
        )
    )
    startup = StartupProfile(
        startup_id="s1",
        name="Acme AI",
        description="AI tooling",
        industries=["AI", "Fintech"],
        stage="Seed",
        country="US",
        region="North America",
        fundraising_amount=1_000_000,
    )
    candidate = Candidate(
        investor=InvestorProfile(
            investor_id="i1",
            name="North Star Ventures",
            description="AI-focused fund",
            industries=["AI", "SaaS"],
            preferred_stages=["Seed"],
            countries=["US"],
            regions=["North America"],
            ticket_min=500_000,
            ticket_max=2_000_000,
        ),
        semantic_similarity=0.8,
        retrieval_rank=1,
    )

    result = service.score_candidate(startup, candidate)

    assert result.industry_match == pytest.approx(0.5)
    assert result.stage_match == pytest.approx(1.0)
    assert result.geo_match == pytest.approx(1.0)
    assert result.ticket_size_fit == pytest.approx(1.0)
    assert result.weighted_score == pytest.approx(0.8)
    assert result.matched_industries == ["AI"]


def test_score_candidate_returns_neutral_scores_for_missing_data() -> None:
    service = ScoringService(ScoringSettings())
    startup = StartupProfile(
        startup_id="s1",
        name="Acme AI",
        description="AI tooling",
    )
    candidate = Candidate(
        investor=InvestorProfile(
            investor_id="i1",
            name="North Star Ventures",
            description="Generalist fund",
        ),
        semantic_similarity=0.4,
        retrieval_rank=1,
    )

    result = service.score_candidate(startup, candidate)

    assert result.industry_match == pytest.approx(0.5)
    assert result.stage_match == pytest.approx(0.5)
    assert result.geo_match == pytest.approx(0.5)
    assert result.ticket_size_fit == pytest.approx(0.5)
    assert "Incomplete stage data" in result.reasons
    assert "Incomplete geographic data" in result.reasons
    assert "Incomplete ticket size data" in result.reasons


def test_score_candidate_rejects_ticket_out_of_range() -> None:
    service = ScoringService(ScoringSettings())
    startup = StartupProfile(
        startup_id="s1",
        name="Acme AI",
        description="AI tooling",
        fundraising_amount=5_000_000,
    )
    candidate = Candidate(
        investor=InvestorProfile(
            investor_id="i1",
            name="North Star Ventures",
            description="Generalist fund",
            ticket_min=250_000,
            ticket_max=1_000_000,
        ),
        semantic_similarity=0.7,
        retrieval_rank=1,
    )

    result = service.score_candidate(startup, candidate)

    assert result.ticket_size_fit == pytest.approx(0.0)


def test_score_candidate_uses_region_when_country_does_not_match() -> None:
    service = ScoringService(ScoringSettings())
    startup = StartupProfile(
        startup_id="s1",
        name="Acme AI",
        description="AI tooling",
        country="Germany",
        region="Europe",
    )
    candidate = Candidate(
        investor=InvestorProfile(
            investor_id="i1",
            name="Euro Ventures",
            description="European early-stage fund",
            countries=["France"],
            regions=["Europe"],
        ),
        semantic_similarity=0.7,
        retrieval_rank=1,
    )

    result = service.score_candidate(startup, candidate)

    assert result.geo_match == pytest.approx(1.0)
