from __future__ import annotations

import pandas as pd
import pytest

from app.services import (
    CSVRowValidationError,
    CSVSchemaError,
    load_investors_csv,
    load_startups_csv,
)


def test_load_startups_csv_returns_typed_profiles(tmp_path) -> None:
    csv_path = tmp_path / "startups.csv"
    pd.DataFrame(
        [
            {
                "startup_id": "s1",
                "name": "Acme AI",
                "description": "AI tooling for diligence workflows",
                "industries": "AI, Fintech",
                "stage": "Seed",
                "country": "US",
                "region": "North America",
                "fundraising_amount": 1500000,
                "currency": "USD",
                "website": "https://acme.example",
            }
        ]
    ).to_csv(csv_path, index=False)

    startups = load_startups_csv(csv_path)

    assert len(startups) == 1
    assert startups[0].startup_id == "s1"
    assert startups[0].industries == ["AI", "Fintech"]
    assert startups[0].fundraising_amount == 1500000


def test_load_investors_csv_returns_typed_profiles(tmp_path) -> None:
    csv_path = tmp_path / "investors.csv"
    pd.DataFrame(
        [
            {
                "investor_id": "i1",
                "name": "North Star Ventures",
                "description": "Early-stage B2B software fund",
                "industries": "AI, SaaS",
                "preferred_stages": "Pre-Seed, Seed",
                "countries": "US, Canada",
                "regions": "North America",
                "ticket_min": 250000,
                "ticket_max": 2000000,
                "currency": "USD",
                "investor_type": "VC",
                "website": "https://northstar.example",
            }
        ]
    ).to_csv(csv_path, index=False)

    investors = load_investors_csv(csv_path)

    assert len(investors) == 1
    assert investors[0].investor_id == "i1"
    assert investors[0].preferred_stages == ["Pre-Seed", "Seed"]
    assert investors[0].countries == ["US", "Canada"]
    assert investors[0].ticket_max == 2000000


def test_load_startups_csv_raises_for_missing_required_columns(tmp_path) -> None:
    csv_path = tmp_path / "startups.csv"
    pd.DataFrame(
        [
            {
                "startup_id": "s1",
                "name": "Acme AI",
            }
        ]
    ).to_csv(csv_path, index=False)

    with pytest.raises(CSVSchemaError):
        load_startups_csv(csv_path)


def test_load_investors_csv_raises_for_invalid_ticket_bounds(tmp_path) -> None:
    csv_path = tmp_path / "investors.csv"
    pd.DataFrame(
        [
            {
                "investor_id": "i1",
                "name": "North Star Ventures",
                "description": "Early-stage B2B software fund",
                "industries": "AI",
                "preferred_stages": "Seed",
                "countries": "US",
                "regions": "North America",
                "ticket_min": 3000000,
                "ticket_max": 1000000,
                "currency": "USD",
            }
        ]
    ).to_csv(csv_path, index=False)

    with pytest.raises(CSVRowValidationError):
        load_investors_csv(csv_path)
