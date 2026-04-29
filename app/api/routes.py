from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_catalog_service, get_matching_service
from app.api.schemas import (
    HealthResponse,
    InvestorListResponse,
    MatchItemResponse,
    MatchRequest,
    MatchResponse,
    StartupListResponse,
)
from app.services import CatalogService, MatchingService


router = APIRouter(tags=["matching"])


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health_check() -> HealthResponse:
    """Simple health endpoint for smoke checks and orchestration probes."""

    return HealthResponse(status="ok")


@router.get("/startups", response_model=StartupListResponse, tags=["catalog"])
def list_startups(
    catalog_service: CatalogService = Depends(get_catalog_service),
) -> StartupListResponse:
    """List sample startups loaded from the default catalog."""

    return StartupListResponse(startups=catalog_service.list_startups())


@router.get("/investors", response_model=InvestorListResponse, tags=["catalog"])
def list_investors(
    catalog_service: CatalogService = Depends(get_catalog_service),
) -> InvestorListResponse:
    """List sample investors loaded from the default catalog."""

    return InvestorListResponse(investors=catalog_service.list_investors())


@router.post("/match", response_model=MatchResponse)
def match_startup_to_investors(
    payload: MatchRequest,
    matching_service: MatchingService = Depends(get_matching_service),
) -> MatchResponse:
    """Run semantic retrieval and reranking for one startup request."""

    matches = matching_service.match_startup(
        startup=payload.startup,
        investors=payload.investors,
        top_k=payload.top_k,
        candidate_pool_size=payload.candidate_pool_size,
    )
    return MatchResponse(matches=_to_match_items(matches))


@router.get("/match/{startup_id}", response_model=MatchResponse)
def match_startup_from_catalog(
    startup_id: str,
    top_k: Annotated[int | None, Query(ge=1)] = None,
    candidate_pool_size: Annotated[int | None, Query(ge=1)] = None,
    catalog_service: CatalogService = Depends(get_catalog_service),
    matching_service: MatchingService = Depends(get_matching_service),
) -> MatchResponse:
    """Match one catalog startup against the default investor catalog."""

    startup = catalog_service.get_startup(startup_id)
    if startup is None:
        raise HTTPException(status_code=404, detail=f"Startup '{startup_id}' not found.")

    matches = matching_service.match_startup(
        startup=startup,
        investors=catalog_service.list_investors(),
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
    )
    return MatchResponse(matches=_to_match_items(matches))


def _to_match_items(matches) -> list[MatchItemResponse]:
    return [
        MatchItemResponse(
            investor_id=match.investor_id,
            investor_name=match.investor_name,
            rank=match.rank,
            final_score=match.score,
            score_breakdown=match.score_breakdown,
            reasons=list(match.score_breakdown.reasons),
        )
        for match in matches
    ]
