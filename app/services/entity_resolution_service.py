from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from app.models import InvestorProfile


class DuplicateCandidate(BaseModel):
    """Potential duplicate investor pair produced by entity resolution."""

    left_investor_id: str
    right_investor_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    blocking_key: str
    reason: str


class EntityResolutionService:
    """Performs lightweight investor deduplication through normalization and blocking."""

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self._similarity_threshold = similarity_threshold

    def find_duplicate_investors(
        self,
        investors: list[InvestorProfile],
    ) -> list[DuplicateCandidate]:
        """Return likely duplicate investor pairs."""

        blocks: dict[str, list[InvestorProfile]] = defaultdict(list)
        for investor in investors:
            for block_key in self._blocking_keys(investor):
                blocks[block_key].append(investor)

        duplicate_pairs: dict[tuple[str, str], DuplicateCandidate] = {}
        for block_key, blocked_investors in blocks.items():
            if len(blocked_investors) < 2:
                continue

            for left, right in combinations(blocked_investors, 2):
                pair_key = tuple(sorted((left.investor_id, right.investor_id)))
                candidate = self._score_duplicate_pair(left, right, block_key)
                if candidate is None:
                    continue
                current_best = duplicate_pairs.get(pair_key)
                if current_best is None or candidate.similarity_score > current_best.similarity_score:
                    duplicate_pairs[pair_key] = candidate

        return sorted(
            duplicate_pairs.values(),
            key=lambda item: (-item.similarity_score, item.left_investor_id, item.right_investor_id),
        )

    def _blocking_keys(self, investor: InvestorProfile) -> set[str]:
        normalized_name = self._normalize_name(investor.name)
        website_domain = self._normalize_domain(investor.website)
        keys = set()

        if normalized_name:
            keys.add(f"name:{normalized_name[:8]}")
        if website_domain:
            keys.add(f"domain:{website_domain}")
        if normalized_name and website_domain:
            keys.add(f"name_domain:{normalized_name[:8]}:{website_domain}")
        return keys

    def _score_duplicate_pair(
        self,
        left: InvestorProfile,
        right: InvestorProfile,
        block_key: str,
    ) -> DuplicateCandidate | None:
        left_name = self._normalize_name(left.name)
        right_name = self._normalize_name(right.name)
        name_similarity = SequenceMatcher(a=left_name, b=right_name).ratio()

        left_domain = self._normalize_domain(left.website)
        right_domain = self._normalize_domain(right.website)
        domain_match = bool(left_domain and right_domain and left_domain == right_domain)

        similarity_score = max(name_similarity, 1.0 if domain_match else 0.0)
        if similarity_score < self._similarity_threshold:
            return None

        if domain_match:
            reason = "Matching website domain"
        else:
            reason = "High normalized name similarity"

        return DuplicateCandidate(
            left_investor_id=left.investor_id,
            right_investor_id=right.investor_id,
            similarity_score=similarity_score,
            blocking_key=block_key,
            reason=reason,
        )

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "", name.casefold())
        return normalized

    @staticmethod
    def _normalize_domain(website: str | None) -> str | None:
        if not website:
            return None
        parsed = urlparse(website if "://" in website else f"https://{website}")
        domain = parsed.netloc.casefold().removeprefix("www.")
        return domain or None
