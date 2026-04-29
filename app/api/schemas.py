from __future__ import annotations

from pydantic import BaseModel, Field

from app.models import InvestorProfile, ScoreBreakdown, StartupProfile


class HealthResponse(BaseModel):
    """Health response for service readiness checks."""

    status: str


class StartupListResponse(BaseModel):
    """Response payload for listing default startup records."""

    startups: list[StartupProfile]


class InvestorListResponse(BaseModel):
    """Response payload for listing default investor records."""

    investors: list[InvestorProfile]


class MatchRequest(BaseModel):
    """API contract for startup-investor matching requests."""

    startup: StartupProfile
    investors: list[InvestorProfile]
    top_k: int | None = Field(default=None, ge=1)
    candidate_pool_size: int | None = Field(default=None, ge=1)


class MatchResponse(BaseModel):
    """API contract for startup-investor matching responses."""

    matches: list["MatchItemResponse"]


class MatchItemResponse(BaseModel):
    """Demo-friendly API representation of a ranked investor match."""

    investor_id: str
    investor_name: str
    rank: int = Field(ge=1)
    final_score: float = Field(ge=0.0)
    score_breakdown: ScoreBreakdown
    reasons: list[str] = Field(default_factory=list)


__all__ = [
    "HealthResponse",
    "InvestorListResponse",
    "MatchItemResponse",
    "MatchRequest",
    "MatchResponse",
    "StartupListResponse",
]
