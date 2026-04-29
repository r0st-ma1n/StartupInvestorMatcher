from __future__ import annotations

from app.config import ScoringSettings
from app.models import Candidate, ScoreBreakdown, StartupProfile


class ScoringService:
    """Applies explainable rule-based scoring on retrieved candidates."""

    def __init__(self, settings: ScoringSettings) -> None:
        self._settings = settings

    def score_candidate(
        self,
        startup: StartupProfile,
        candidate: Candidate,
    ) -> ScoreBreakdown:
        """Score one retrieved investor candidate for a given startup."""

        investor = candidate.investor
        matched_industries = self._matched_industries(startup.industries, investor.industries)

        semantic_similarity = candidate.semantic_similarity
        industry_match = self._score_industry_match(startup.industries, investor.industries)
        stage_match = self._score_stage_match(startup.stage, investor.preferred_stages)
        geo_match = self._score_geo_match(
            startup.country,
            startup.region,
            investor.countries,
            investor.regions,
        )
        ticket_size_fit = self._score_ticket_size_fit(
            startup.fundraising_amount,
            investor.ticket_min,
            investor.ticket_max,
        )

        weighted_score = (
            semantic_similarity * self._settings.semantic_similarity_weight
            + industry_match * self._settings.industry_match_weight
            + stage_match * self._settings.stage_match_weight
            + geo_match * self._settings.geo_match_weight
            + ticket_size_fit * self._settings.ticket_size_fit_weight
        )

        return ScoreBreakdown(
            semantic_similarity=semantic_similarity,
            industry_match=industry_match,
            stage_match=stage_match,
            geo_match=geo_match,
            ticket_size_fit=ticket_size_fit,
            weighted_score=weighted_score,
            matched_industries=matched_industries,
            reasons=self._build_reasons(
                semantic_similarity=semantic_similarity,
                matched_industries=matched_industries,
                stage_match=stage_match,
                geo_match=geo_match,
                ticket_size_fit=ticket_size_fit,
            ),
        )

    @staticmethod
    def _matched_industries(
        startup_industries: list[str],
        investor_industries: list[str],
    ) -> list[str]:
        investor_lookup = {industry.casefold(): industry for industry in investor_industries}
        matches: list[str] = []
        for industry in startup_industries:
            matched = investor_lookup.get(industry.casefold())
            if matched:
                matches.append(matched)
        return matches

    def _score_industry_match(
        self,
        startup_industries: list[str],
        investor_industries: list[str],
    ) -> float:
        if not startup_industries or not investor_industries:
            return 0.5
        matched = self._matched_industries(startup_industries, investor_industries)
        return len(matched) / len(startup_industries)

    @staticmethod
    def _score_stage_match(startup_stage: str | None, preferred_stages: list[str]) -> float:
        if not startup_stage or not preferred_stages:
            return 0.5
        normalized_stage = startup_stage.casefold()
        return 1.0 if any(stage.casefold() == normalized_stage for stage in preferred_stages) else 0.0

    @staticmethod
    def _score_geo_match(
        startup_country: str | None,
        startup_region: str | None,
        investor_countries: list[str],
        investor_regions: list[str],
    ) -> float:
        if not startup_country and not startup_region:
            return 0.5
        if not investor_countries and not investor_regions:
            return 0.5

        if startup_country and any(country.casefold() == startup_country.casefold() for country in investor_countries):
            return 1.0
        if startup_region and any(region.casefold() == startup_region.casefold() for region in investor_regions):
            return 1.0
        return 0.0

    @staticmethod
    def _score_ticket_size_fit(
        fundraising_amount: float | None,
        ticket_min: float | None,
        ticket_max: float | None,
    ) -> float:
        if fundraising_amount is None:
            return 0.5
        if ticket_min is None and ticket_max is None:
            return 0.5
        if ticket_min is not None and fundraising_amount < ticket_min:
            return 0.0
        if ticket_max is not None and fundraising_amount > ticket_max:
            return 0.0
        if ticket_min is None or ticket_max is None:
            return 0.5
        return 1.0

    @staticmethod
    def _build_reasons(
        *,
        semantic_similarity: float,
        matched_industries: list[str],
        stage_match: float,
        geo_match: float,
        ticket_size_fit: float,
    ) -> list[str]:
        reasons: list[str] = []
        if semantic_similarity >= 0.75:
            reasons.append("Strong semantic similarity")
        elif semantic_similarity >= 0.5:
            reasons.append("Moderate semantic similarity")

        if matched_industries:
            reasons.append(f"Industry overlap: {', '.join(matched_industries)}")
        if stage_match == 1.0:
            reasons.append("Stage preference match")
        elif stage_match == 0.5:
            reasons.append("Incomplete stage data")

        if geo_match == 1.0:
            reasons.append("Geographic focus match")
        elif geo_match == 0.5:
            reasons.append("Incomplete geographic data")

        if ticket_size_fit == 1.0:
            reasons.append("Ticket size fit")
        elif ticket_size_fit == 0.5:
            reasons.append("Incomplete ticket size data")

        return reasons
