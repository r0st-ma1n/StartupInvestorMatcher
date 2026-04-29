from __future__ import annotations

import pytest

from app.models import InvestorProfile
from app.services import EntityResolutionService


def test_find_duplicate_investors_matches_by_normalized_name() -> None:
    service = EntityResolutionService(similarity_threshold=0.85)
    investors = [
        InvestorProfile(
            investor_id="i1",
            name="North Star Ventures",
            description="Fund",
        ),
        InvestorProfile(
            investor_id="i2",
            name="North-Star Ventures",
            description="Fund duplicate",
        ),
    ]

    duplicates = service.find_duplicate_investors(investors)

    assert len(duplicates) == 1
    assert duplicates[0].left_investor_id == "i1"
    assert duplicates[0].right_investor_id == "i2"
    assert duplicates[0].similarity_score == pytest.approx(1.0)


def test_find_duplicate_investors_matches_by_website_domain() -> None:
    service = EntityResolutionService(similarity_threshold=0.95)
    investors = [
        InvestorProfile(
            investor_id="i1",
            name="NSV Capital",
            description="Fund",
            website="https://northstar.vc",
        ),
        InvestorProfile(
            investor_id="i2",
            name="North Star Ventures",
            description="Fund duplicate",
            website="http://www.northstar.vc/team",
        ),
    ]

    duplicates = service.find_duplicate_investors(investors)

    assert len(duplicates) == 1
    assert duplicates[0].reason == "Matching website domain"


def test_find_duplicate_investors_ignores_dissimilar_entities() -> None:
    service = EntityResolutionService(similarity_threshold=0.9)
    investors = [
        InvestorProfile(
            investor_id="i1",
            name="North Star Ventures",
            description="Fund",
        ),
        InvestorProfile(
            investor_id="i2",
            name="Blue Ocean Capital",
            description="Another fund",
        ),
    ]

    duplicates = service.find_duplicate_investors(investors)

    assert duplicates == []
