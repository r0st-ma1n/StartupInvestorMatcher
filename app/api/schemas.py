from __future__ import annotations

from pydantic import BaseModel, Field

from app.models import InvestorProfile, MatchResult, StartupProfile


class MatchRequest(BaseModel):
    """API contract for startup-investor matching requests."""

    startup: StartupProfile
    investors: list[InvestorProfile]
    top_k: int | None = Field(default=None, ge=1)
    candidate_pool_size: int | None = Field(default=None, ge=1)


class MatchResponse(BaseModel):
    """API contract for startup-investor matching responses."""

    matches: list[MatchResult]

__all__ = ["MatchRequest", "MatchResponse"]
