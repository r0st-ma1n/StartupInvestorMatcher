from __future__ import annotations

import numpy as np
import pytest

from app.config import RetrievalSettings
from app.models import InvestorProfile, StartupProfile
from app.services import EmbeddingShapeError, RetrievalError, RetrievalService


def test_compute_similarity_matrix_returns_expected_shape() -> None:
    service = RetrievalService(RetrievalSettings())

    startup_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    investor_embeddings = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    similarity_matrix = service.compute_similarity_matrix(
        startup_embeddings,
        investor_embeddings,
    )

    assert similarity_matrix.shape == (2, 2)
    assert similarity_matrix[0, 0] == pytest.approx(1.0)
    assert similarity_matrix[1, 0] == pytest.approx(0.0)


def test_retrieve_top_k_candidates_orders_by_similarity() -> None:
    service = RetrievalService(RetrievalSettings(default_top_k=2))
    startup = StartupProfile(
        startup_id="s1",
        name="Acme AI",
        description="AI tooling",
    )
    investors = [
        InvestorProfile(
            investor_id="i1",
            name="Alpha Ventures",
            description="Generalist fund",
        ),
        InvestorProfile(
            investor_id="i2",
            name="Beta Capital",
            description="AI specialist",
        ),
        InvestorProfile(
            investor_id="i3",
            name="Gamma Fund",
            description="Climate investor",
        ),
    ]
    startup_embedding = np.array([1.0, 0.0], dtype=np.float32)
    investor_embeddings = np.array(
        [
            [0.8, 0.6],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    candidates = service.retrieve_top_k_candidates(
        startup=startup,
        startup_embedding=startup_embedding,
        investors=investors,
        investor_embeddings=investor_embeddings,
    )

    assert len(candidates) == 2
    assert candidates[0].investor.investor_id == "i2"
    assert candidates[0].retrieval_rank == 1
    assert candidates[1].investor.investor_id == "i1"
    assert candidates[1].retrieval_rank == 2


def test_retrieve_top_k_candidates_validates_investor_count() -> None:
    service = RetrievalService(RetrievalSettings())
    startup = StartupProfile(
        startup_id="s1",
        name="Acme AI",
        description="AI tooling",
    )
    investors = [
        InvestorProfile(
            investor_id="i1",
            name="Alpha Ventures",
            description="Generalist fund",
        )
    ]
    startup_embedding = np.array([1.0, 0.0], dtype=np.float32)
    investor_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(EmbeddingShapeError):
        service.retrieve_top_k_candidates(
            startup=startup,
            startup_embedding=startup_embedding,
            investors=investors,
            investor_embeddings=investor_embeddings,
        )


def test_retrieve_top_k_candidates_rejects_invalid_top_k() -> None:
    service = RetrievalService(RetrievalSettings())
    startup = StartupProfile(
        startup_id="s1",
        name="Acme AI",
        description="AI tooling",
    )
    investors = [
        InvestorProfile(
            investor_id="i1",
            name="Alpha Ventures",
            description="Generalist fund",
        )
    ]
    startup_embedding = np.array([1.0, 0.0], dtype=np.float32)
    investor_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(RetrievalError):
        service.retrieve_top_k_candidates(
            startup=startup,
            startup_embedding=startup_embedding,
            investors=investors,
            investor_embeddings=investor_embeddings,
            top_k=0,
        )
