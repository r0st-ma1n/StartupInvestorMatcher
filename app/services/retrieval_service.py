from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.config import RetrievalSettings
from app.models import Candidate, InvestorProfile, StartupProfile


class RetrievalError(ValueError):
    """Base error for candidate retrieval failures."""


class EmbeddingShapeError(RetrievalError):
    """Raised when embedding inputs do not have compatible shapes."""


class RetrievalService:
    """Performs semantic candidate retrieval using cosine similarity."""

    def __init__(self, settings: RetrievalSettings) -> None:
        self._settings = settings

    def compute_similarity_matrix(
        self,
        startup_embeddings: np.ndarray,
        investor_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute startup-to-investor cosine similarity scores."""

        startup_matrix = self._as_2d_matrix(startup_embeddings, "startup_embeddings")
        investor_matrix = self._as_2d_matrix(investor_embeddings, "investor_embeddings")

        if startup_matrix.shape[1] != investor_matrix.shape[1]:
            raise EmbeddingShapeError(
                "Embedding dimensions must match for cosine similarity: "
                f"{startup_matrix.shape[1]} != {investor_matrix.shape[1]}"
            )

        return cosine_similarity(startup_matrix, investor_matrix).astype(np.float32)

    def retrieve_top_k_candidates(
        self,
        startup: StartupProfile,
        startup_embedding: np.ndarray,
        investors: Sequence[InvestorProfile],
        investor_embeddings: np.ndarray,
        top_k: int | None = None,
    ) -> list[Candidate]:
        """Return the top-k investors for one startup by semantic similarity."""

        if not investors:
            return []

        investor_matrix = self._as_2d_matrix(investor_embeddings, "investor_embeddings")
        if investor_matrix.shape[0] != len(investors):
            raise EmbeddingShapeError(
                "Number of investor embeddings must match number of investors: "
                f"{investor_matrix.shape[0]} != {len(investors)}"
            )

        similarity_scores = self.compute_similarity_matrix(
            self._as_2d_matrix(startup_embedding, "startup_embedding"),
            investor_matrix,
        )[0]

        requested_top_k = self._settings.default_top_k if top_k is None else top_k
        if requested_top_k < 1:
            raise RetrievalError("top_k must be greater than 0.")

        top_indices = np.argsort(-similarity_scores)[:requested_top_k]
        candidates: list[Candidate] = []
        for rank, index in enumerate(top_indices, start=1):
            candidates.append(
                Candidate(
                    investor=investors[int(index)],
                    semantic_similarity=float(similarity_scores[int(index)]),
                    retrieval_rank=rank,
                )
            )
        return candidates

    @staticmethod
    def _as_2d_matrix(embeddings: np.ndarray, name: str) -> np.ndarray:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise EmbeddingShapeError(f"{name} must be a 2D matrix or 1D vector.")
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise EmbeddingShapeError(f"{name} cannot be empty.")
        return matrix
