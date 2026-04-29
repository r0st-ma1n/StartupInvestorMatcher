from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.models import InvestorProfile
from app.services.embedding_service import EmbeddingService


@dataclass(frozen=True)
class InvestorEmbeddingIndex:
    """In-memory representation of a precomputed investor embedding artifact."""

    investor_ids: list[str]
    embeddings: np.ndarray
    model_name: str
    source_hash: str
    profile_hashes: dict[str, str]


class InvestorIndexService:
    """Builds, loads, and resolves precomputed investor embeddings."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self._embedding_service = embedding_service

    def build_index(
        self,
        investors: list[InvestorProfile],
        output_path: str | Path,
        source_hash: str,
    ) -> InvestorEmbeddingIndex:
        """Build and persist an investor embedding artifact."""

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        investor_ids = [investor.investor_id for investor in investors]
        embeddings = self._embedding_service.embed_investors(investors)
        profile_hashes = {
            investor.investor_id: self._profile_hash(investor, self._embedding_service)
            for investor in investors
        }

        np.savez_compressed(
            output,
            investor_ids=np.array(investor_ids, dtype=str),
            embeddings=embeddings.astype(np.float32),
            model_name=np.array(self._embedding_service.model_name, dtype=str),
            source_hash=np.array(source_hash, dtype=str),
            profile_hashes=np.array(
                [profile_hashes[investor_id] for investor_id in investor_ids],
                dtype=str,
            ),
        )
        return InvestorEmbeddingIndex(
            investor_ids=investor_ids,
            embeddings=embeddings.astype(np.float32),
            model_name=self._embedding_service.model_name,
            source_hash=source_hash,
            profile_hashes=profile_hashes,
        )

    def load_index(self, path: str | Path) -> InvestorEmbeddingIndex | None:
        """Load a precomputed investor embedding artifact if it exists."""

        artifact_path = Path(path)
        if not artifact_path.exists():
            return None

        with np.load(artifact_path, allow_pickle=False) as payload:
            investor_ids = payload["investor_ids"].astype(str).tolist()
            embeddings = payload["embeddings"].astype(np.float32)
            model_name = str(payload["model_name"].item())
            source_hash = str(payload["source_hash"].item())
            profile_hash_values = payload["profile_hashes"].astype(str).tolist()

        return InvestorEmbeddingIndex(
            investor_ids=investor_ids,
            embeddings=embeddings,
            model_name=model_name,
            source_hash=source_hash,
            profile_hashes=dict(zip(investor_ids, profile_hash_values, strict=True)),
        )

    def resolve_embeddings(
        self,
        investors: list[InvestorProfile],
        index: InvestorEmbeddingIndex | None,
    ) -> np.ndarray | None:
        """Return precomputed embeddings when the artifact matches the investors."""

        if index is None:
            return None
        if index.model_name != self._embedding_service.model_name:
            return None

        resolved_rows: list[np.ndarray] = []
        id_to_position = {investor_id: idx for idx, investor_id in enumerate(index.investor_ids)}
        for investor in investors:
            position = id_to_position.get(investor.investor_id)
            if position is None:
                return None
            expected_hash = index.profile_hashes.get(investor.investor_id)
            actual_hash = self._profile_hash(investor, self._embedding_service)
            if expected_hash != actual_hash:
                return None
            resolved_rows.append(index.embeddings[position])

        return np.asarray(resolved_rows, dtype=np.float32)

    @staticmethod
    def compute_source_hash(path: str | Path) -> str:
        """Hash the raw source file to version the artifact against its inputs."""

        content = Path(path).read_bytes()
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _profile_hash(investor: InvestorProfile, embedding_service: EmbeddingService) -> str:
        text = embedding_service.format_investor_text(investor)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
