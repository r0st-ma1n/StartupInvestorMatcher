from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_matching_service
from app.api.schemas import MatchRequest, MatchResponse
from app.services import MatchingService


router = APIRouter(tags=["matching"])


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
    return MatchResponse(matches=matches)
