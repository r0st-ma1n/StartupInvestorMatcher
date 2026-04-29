from __future__ import annotations

from pathlib import Path

from app.models import InvestorProfile, StartupProfile
from app.services.data_loader import load_investors_csv, load_startups_csv


class CatalogService:
    """Loads and serves the default startup and investor catalogs."""

    def __init__(self, startups_path: str | Path, investors_path: str | Path) -> None:
        self._startups_path = Path(startups_path)
        self._investors_path = Path(investors_path)
        self._startups: list[StartupProfile] | None = None
        self._investors: list[InvestorProfile] | None = None

    def list_startups(self) -> list[StartupProfile]:
        if self._startups is None:
            self._startups = load_startups_csv(self._startups_path)
        return list(self._startups)

    def list_investors(self) -> list[InvestorProfile]:
        if self._investors is None:
            self._investors = load_investors_csv(self._investors_path)
        return list(self._investors)

    def get_startup(self, startup_id: str) -> StartupProfile | None:
        for startup in self.list_startups():
            if startup.startup_id == startup_id:
                return startup
        return None
