from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from app.models import InvestorProfile, StartupProfile


class CSVLoaderError(ValueError):
    """Base error for CSV ingestion failures."""


class CSVSchemaError(CSVLoaderError):
    """Raised when a CSV file does not match the expected schema."""


class CSVRowValidationError(CSVLoaderError):
    """Raised when a CSV row cannot be converted into a domain model."""


STARTUP_REQUIRED_COLUMNS = {
    "startup_id",
    "name",
    "description",
    "industries",
    "stage",
    "country",
    "region",
    "fundraising_amount",
    "currency",
}

INVESTOR_REQUIRED_COLUMNS = {
    "investor_id",
    "name",
    "description",
    "industries",
    "preferred_stages",
    "countries",
    "regions",
    "ticket_min",
    "ticket_max",
    "currency",
}


def load_startups_csv(path: str | Path) -> list[StartupProfile]:
    """Load startups from CSV into typed domain models."""

    dataframe = _read_csv(path)
    _validate_required_columns(dataframe, STARTUP_REQUIRED_COLUMNS, path)

    startups: list[StartupProfile] = []
    for row_index, row in dataframe.iterrows():
        csv_row_number = row_index + 2
        try:
            startups.append(
                StartupProfile(
                    startup_id=_require_string(row, "startup_id", csv_row_number),
                    name=_require_string(row, "name", csv_row_number),
                    description=_require_string(row, "description", csv_row_number),
                    industries=row.get("industries"),
                    stage=_optional_string(row.get("stage")),
                    country=_optional_string(row.get("country")),
                    region=_optional_string(row.get("region")),
                    fundraising_amount=_optional_float(row.get("fundraising_amount")),
                    currency=_optional_string(row.get("currency")) or "USD",
                    website=_optional_string(row.get("website")),
                )
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise CSVRowValidationError(
                f"Invalid startup row {csv_row_number}: {exc}"
            ) from exc

    return startups


def load_investors_csv(path: str | Path) -> list[InvestorProfile]:
    """Load investors from CSV into typed domain models."""

    dataframe = _read_csv(path)
    _validate_required_columns(dataframe, INVESTOR_REQUIRED_COLUMNS, path)

    investors: list[InvestorProfile] = []
    for row_index, row in dataframe.iterrows():
        csv_row_number = row_index + 2
        try:
            investors.append(
                InvestorProfile(
                    investor_id=_require_string(row, "investor_id", csv_row_number),
                    name=_require_string(row, "name", csv_row_number),
                    description=_require_string(row, "description", csv_row_number),
                    industries=row.get("industries"),
                    preferred_stages=row.get("preferred_stages"),
                    countries=row.get("countries"),
                    regions=row.get("regions"),
                    ticket_min=_optional_float(row.get("ticket_min")),
                    ticket_max=_optional_float(row.get("ticket_max")),
                    currency=_optional_string(row.get("currency")) or "USD",
                    investor_type=_optional_string(row.get("investor_type")),
                    website=_optional_string(row.get("website")),
                )
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise CSVRowValidationError(
                f"Invalid investor row {csv_row_number}: {exc}"
            ) from exc

    return investors


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    dataframe = pd.read_csv(csv_path)
    if dataframe.empty:
        raise CSVSchemaError(f"CSV file is empty: {csv_path}")
    return dataframe


def _validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    path: str | Path,
) -> None:
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        raise CSVSchemaError(
            f"Missing required columns in {Path(path)}: {', '.join(missing_columns)}"
        )


def _require_string(row: pd.Series, column: str, csv_row_number: int) -> str:
    value = _optional_string(row.get(column))
    if value is None:
        raise ValueError(f"Column '{column}' is required at row {csv_row_number}.")
    return value


def _optional_string(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    return float(value)
