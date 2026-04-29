from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.dependencies import get_catalog_service, get_matching_service
from app.main import app
from app.models import Candidate, InvestorProfile, MatchResult, ScoreBreakdown, StartupProfile


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


class StubCatalogService:
    def __init__(self) -> None:
        self.startups = [
            StartupProfile(
                startup_id="s1",
                name="Acme AI",
                description="AI tooling for diligence",
                industries=["AI"],
                stage="Seed",
                country="US",
                region="North America",
                fundraising_amount=1000000,
                currency="USD",
            )
        ]
        self.investors = [
            InvestorProfile(
                investor_id="i1",
                name="North Star Ventures",
                description="AI-focused fund",
                industries=["AI"],
                preferred_stages=["Seed"],
                countries=["US"],
                regions=["North America"],
                ticket_min=250000,
                ticket_max=2000000,
                currency="USD",
            )
        ]

    def list_startups(self):
        return list(self.startups)

    def list_investors(self):
        return list(self.investors)

    def get_startup(self, startup_id: str):
        for startup in self.startups:
            if startup.startup_id == startup_id:
                return startup
        return None


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_startups_endpoint_returns_catalog_records() -> None:
    app.dependency_overrides[get_catalog_service] = lambda: StubCatalogService()
    client = TestClient(app)

    response = client.get("/startups")

    assert response.status_code == 200
    body = response.json()
    assert len(body["startups"]) == 1
    assert body["startups"][0]["startup_id"] == "s1"

    app.dependency_overrides.clear()


def test_investors_endpoint_returns_catalog_records() -> None:
    app.dependency_overrides[get_catalog_service] = lambda: StubCatalogService()
    client = TestClient(app)

    response = client.get("/investors")

    assert response.status_code == 200
    body = response.json()
    assert len(body["investors"]) == 1
    assert body["investors"][0]["investor_id"] == "i1"

    app.dependency_overrides.clear()


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
    assert body["matches"][0]["final_score"] == 0.9505
    assert body["matches"][0]["reasons"] == ["Strong semantic similarity", "Stage preference match"]
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


def test_match_by_startup_id_returns_matches() -> None:
    stub_service = StubMatchingService()
    app.dependency_overrides[get_matching_service] = lambda: stub_service
    app.dependency_overrides[get_catalog_service] = lambda: StubCatalogService()
    client = TestClient(app)

    response = client.get("/match/s1?top_k=1&candidate_pool_size=2")

    assert response.status_code == 200
    body = response.json()
    assert len(body["matches"]) == 1
    assert body["matches"][0]["investor_name"] == "North Star Ventures"
    assert body["matches"][0]["score_breakdown"]["semantic_similarity"] == 0.91

    app.dependency_overrides.clear()


def test_match_by_startup_id_returns_404_for_missing_startup() -> None:
    app.dependency_overrides[get_matching_service] = lambda: StubMatchingService()
    app.dependency_overrides[get_catalog_service] = lambda: StubCatalogService()
    client = TestClient(app)

    response = client.get("/match/unknown")

    assert response.status_code == 404

    app.dependency_overrides.clear()
