from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",")]
        return [item for item in parts if item]
    if isinstance(value, (list, tuple, set)):
        normalized: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized
    raise TypeError("Expected a string or iterable of strings.")


class DomainModel(BaseModel):
    """Base model with production-friendly defaults for domain entities."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")


class StartupProfile(DomainModel):
    """Canonical startup record used by matching services."""

    startup_id: str
    name: str
    description: str
    industries: list[str] = Field(default_factory=list)
    stage: str | None = None
    country: str | None = None
    region: str | None = None
    fundraising_amount: float | None = Field(default=None, ge=0)
    currency: str = "USD"
    website: str | None = None

    @field_validator("industries", mode="before")
    @classmethod
    def normalize_industries(cls, value: Any) -> list[str]:
        return _normalize_string_list(value)


class InvestorProfile(DomainModel):
    """Canonical investor record used by retrieval and scoring services."""

    investor_id: str
    name: str
    description: str
    industries: list[str] = Field(default_factory=list)
    preferred_stages: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    ticket_min: float | None = Field(default=None, ge=0)
    ticket_max: float | None = Field(default=None, ge=0)
    currency: str = "USD"
    investor_type: str | None = None
    website: str | None = None

    @field_validator("industries", "preferred_stages", "countries", "regions", mode="before")
    @classmethod
    def normalize_string_collections(cls, value: Any) -> list[str]:
        return _normalize_string_list(value)

    @model_validator(mode="after")
    def validate_ticket_bounds(self) -> "InvestorProfile":
        if (
            self.ticket_min is not None
            and self.ticket_max is not None
            and self.ticket_min > self.ticket_max
        ):
            raise ValueError("ticket_min cannot be greater than ticket_max.")
        return self


class Candidate(DomainModel):
    """Retrieved investor candidate before business-rule reranking."""

    investor: InvestorProfile
    semantic_similarity: float = Field(ge=-1.0, le=1.0)
    retrieval_rank: int = Field(ge=1)


class ScoreBreakdown(DomainModel):
    """Explainable component scores for a startup-investor pair."""

    semantic_similarity: float = Field(ge=-1.0, le=1.0)
    industry_match: float = Field(ge=0.0, le=1.0)
    stage_match: float = Field(ge=0.0, le=1.0)
    geo_match: float = Field(ge=0.0, le=1.0)
    ticket_size_fit: float = Field(ge=0.0, le=1.0)
    weighted_score: float = Field(ge=0.0)
    matched_industries: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    @field_validator("matched_industries", "reasons", mode="before")
    @classmethod
    def normalize_lists(cls, value: Any) -> list[str]:
        return _normalize_string_list(value)


class MatchResult(DomainModel):
    """Final ranked match returned by the matching service or API."""

    startup_id: str
    investor_id: str
    investor_name: str
    rank: int = Field(ge=1)
    score: float = Field(ge=0.0)
    candidate: Candidate
    score_breakdown: ScoreBreakdown
