from __future__ import annotations

import numpy as np

from app.config import EmbeddingSettings
from app.models import InvestorProfile, StartupProfile
from app.services import EmbeddingService


class StubEncoder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        sentences,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        self.calls.append(
            {
                "sentences": list(sentences),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "convert_to_numpy": convert_to_numpy,
            }
        )
        return np.array([[1.0, 0.0], [0.0, 1.0]][: len(sentences)], dtype=np.float32)


def test_embed_texts_uses_encoder_settings() -> None:
    encoder = StubEncoder()
    service = EmbeddingService(
        settings=EmbeddingSettings(batch_size=8, normalize_embeddings=False),
        encoder_factory=lambda _: encoder,
    )

    embeddings = service.embed_texts(["  alpha  beta  ", "gamma"])

    assert embeddings.shape == (2, 2)
    assert encoder.calls[0]["sentences"] == ["alpha beta", "gamma"]
    assert encoder.calls[0]["batch_size"] == 8
    assert encoder.calls[0]["normalize_embeddings"] is False
    assert encoder.calls[0]["convert_to_numpy"] is True


def test_embed_startups_formats_profile_text() -> None:
    encoder = StubEncoder()
    service = EmbeddingService(
        settings=EmbeddingSettings(),
        encoder_factory=lambda _: encoder,
    )
    startups = [
        StartupProfile(
            startup_id="s1",
            name="Acme AI",
            description="AI tooling for diligence workflows",
            industries=["AI", "Fintech"],
            stage="Seed",
            country="US",
            region="North America",
        )
    ]

    service.embed_startups(startups)

    assert encoder.calls[0]["sentences"] == [
        "Acme AI AI tooling for diligence workflows Industries: AI, Fintech Stage: Seed Country: US Region: North America"
    ]


def test_embed_investors_formats_profile_text() -> None:
    encoder = StubEncoder()
    service = EmbeddingService(
        settings=EmbeddingSettings(),
        encoder_factory=lambda _: encoder,
    )
    investors = [
        InvestorProfile(
            investor_id="i1",
            name="North Star Ventures",
            description="Early-stage B2B software fund",
            industries=["AI", "SaaS"],
            preferred_stages=["Pre-Seed", "Seed"],
            countries=["US", "Canada"],
            regions=["North America"],
        )
    ]

    service.embed_investors(investors)

    assert encoder.calls[0]["sentences"] == [
        "North Star Ventures Early-stage B2B software fund Industries: AI, SaaS Preferred stages: Pre-Seed, Seed Countries: US, Canada Regions: North America"
    ]


def test_embed_texts_returns_empty_matrix_for_empty_input() -> None:
    service = EmbeddingService(
        settings=EmbeddingSettings(),
        encoder_factory=lambda _: StubEncoder(),
    )

    embeddings = service.embed_texts([])

    assert embeddings.shape == (0, 0)
