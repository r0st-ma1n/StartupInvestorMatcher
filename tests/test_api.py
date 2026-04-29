from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.dependencies import get_matching_service
from app.main import app
from app.models import Candidate, InvestorProfile, MatchResult, ScoreBreakdown


class StubMatchingService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def match_startup(self, startup, investors, top_k=None, candidate_pool_size=None):
        self.calls.append(
            {
                "startup": startup,
                "investors": investors,
                "top_k": top_k,
                "candidate_pool_size": candidate_pool_size,
            }
        )
        candidate = Candidate(
            investor=InvestorProfile(
                investor_id="i1",
                name="North Star Ventures",
                description="AI-focused fund",
                industries=["AI"],
                preferred_stages=["Seed"],
                countries=["US"],
                regions=["North America"],
                ticket_min=250000,
                ticket_max=2000000,
            ),
            semantic_similarity=0.91,
            retrieval_rank=1,
        )
        score_breakdown = ScoreBreakdown(
            semantic_similarity=0.91,
            industry_match=1.0,
            stage_match=1.0,
            geo_match=1.0,
            ticket_size_fit=1.0,
            weighted_score=0.9505,
            matched_industries=["AI"],
            reasons=["Strong semantic similarity", "Stage preference match"],
        )
        return [
            MatchResult(
                startup_id="s1",
                investor_id="i1",
                investor_name="North Star Ventures",
                rank=1,
                score=0.9505,
                candidate=candidate,
                score_breakdown=score_breakdown,
            )
        ]


def test_match_endpoint_returns_matches() -> None:
    stub_service = StubMatchingService()
    app.dependency_overrides[get_matching_service] = lambda: stub_service
    client = TestClient(app)

    response = client.post(
        "/match",
        json={
            "startup": {
                "startup_id": "s1",
                "name": "Acme AI",
                "description": "AI tooling for diligence",
                "industries": ["AI"],
                "stage": "Seed",
                "country": "US",
                "region": "North America",
                "fundraising_amount": 1000000,
                "currency": "USD",
            },
            "investors": [
                {
                    "investor_id": "i1",
                    "name": "North Star Ventures",
                    "description": "AI-focused fund",
                    "industries": ["AI"],
                    "preferred_stages": ["Seed"],
                    "countries": ["US"],
                    "regions": ["North America"],
                    "ticket_min": 250000,
                    "ticket_max": 2000000,
                    "currency": "USD",
                }
            ],
            "top_k": 1,
            "candidate_pool_size": 5,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["matches"]) == 1
    assert body["matches"][0]["investor_id"] == "i1"
    assert stub_service.calls[0]["top_k"] == 1
    assert stub_service.calls[0]["candidate_pool_size"] == 5

    app.dependency_overrides.clear()


def test_match_endpoint_validates_top_k() -> None:
    stub_service = StubMatchingService()
    app.dependency_overrides[get_matching_service] = lambda: stub_service
    client = TestClient(app)

    response = client.post(
        "/match",
        json={
            "startup": {
                "startup_id": "s1",
                "name": "Acme AI",
                "description": "AI tooling for diligence",
            },
            "investors": [],
            "top_k": 0,
        },
    )

    assert response.status_code == 422

    app.dependency_overrides.clear()
