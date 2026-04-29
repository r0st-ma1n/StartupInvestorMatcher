from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path

import numpy as np

from app.config import EmbeddingSettings, RetrievalSettings, ScoringSettings
from app.models import InvestorProfile, StartupProfile
from app.services import (
    EmbeddingService,
    InvestorIndexService,
    MatchingService,
    RetrievalService,
    ScoringService,
)


class StubEncoder:
    def encode(
        self,
        sentences,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        vectors = []
        for index, _ in enumerate(sentences, start=1):
            vectors.append([float(index), float(index) / 10.0])
        return np.array(vectors, dtype=np.float32)


class CountingEmbeddingService:
    def __init__(self, startup_embeddings: np.ndarray, investor_embeddings: np.ndarray) -> None:
        self._startup_embeddings = startup_embeddings
        self._investor_embeddings = investor_embeddings
        self.investor_embed_calls = 0

    def embed_startups(self, startups):
        return self._startup_embeddings

    def embed_investors(self, investors):
        self.investor_embed_calls += 1
        return self._investor_embeddings


def test_build_and_load_investor_index() -> None:
    embedding_service = EmbeddingService(
        EmbeddingSettings(),
        encoder_factory=lambda _: StubEncoder(),
    )
    index_service = InvestorIndexService(embedding_service)
    investors = [
        InvestorProfile(
            investor_id="i1",
            name="North Star Ventures",
            description="AI-focused seed fund",
            industries=["AI"],
        ),
        InvestorProfile(
            investor_id="i2",
            name="Euro Climate Capital",
            description="Climate software investor",
            industries=["Climate"],
        ),
    ]
    artifact_dir = _workspace_temp_dir()
    source_file = artifact_dir / "investors.csv"
    source_file.write_text("investor_id,name\n", encoding="utf-8")
    source_hash = index_service.compute_source_hash(source_file)
    artifact_path = artifact_dir / "investor_index.npz"

    built = index_service.build_index(investors, artifact_path, source_hash=source_hash)
    loaded = index_service.load_index(artifact_path)

    assert loaded is not None
    assert loaded.investor_ids == built.investor_ids
    assert loaded.model_name == embedding_service.model_name
    assert loaded.source_hash == source_hash
    assert loaded.embeddings.shape == (2, 2)
    assert loaded.profile_hashes["i1"] == hashlib.sha256(
        embedding_service.format_investor_text(investors[0]).encode("utf-8")
    ).hexdigest()
    _cleanup_temp_dir(artifact_dir)


def test_matching_service_uses_precomputed_investor_embeddings_when_available() -> None:
    startup = StartupProfile(
        startup_id="s1",
        name="Acme AI",
        description="AI tooling",
        industries=["AI"],
        stage="Seed",
        country="US",
        region="North America",
        fundraising_amount=1_000_000,
    )
    investors = [
        InvestorProfile(
            investor_id="i1",
            name="North Star Ventures",
            description="AI-focused seed fund",
            industries=["AI"],
            preferred_stages=["Seed"],
            countries=["US"],
            regions=["North America"],
            ticket_min=250000,
            ticket_max=2000000,
        )
    ]
    real_embedding_service = EmbeddingService(
        EmbeddingSettings(),
        encoder_factory=lambda _: StubEncoder(),
    )
    index_service = InvestorIndexService(real_embedding_service)
    artifact_dir = _workspace_temp_dir()
    artifact = index_service.build_index(
        investors=investors,
        output_path=artifact_dir / "investor_index.npz",
        source_hash="sample-hash",
    )
    counting_service = CountingEmbeddingService(
        startup_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        investor_embeddings=np.array([[0.0, 1.0]], dtype=np.float32),
    )
    matching_service = MatchingService(
        embedding_service=counting_service,
        retrieval_service=RetrievalService(RetrievalSettings(default_top_k=1, candidate_pool_size=1)),
        retrieval_settings=RetrievalSettings(default_top_k=1, candidate_pool_size=1),
        scoring_service=ScoringService(ScoringSettings()),
        investor_index_service=index_service,
        investor_index=artifact,
    )

    matches = matching_service.match_startup(startup, investors, top_k=1, candidate_pool_size=1)

    assert len(matches) == 1
    assert counting_service.investor_embed_calls == 0
    _cleanup_temp_dir(artifact_dir)


def test_matching_service_falls_back_when_index_does_not_match() -> None:
    startup = StartupProfile(
        startup_id="s1",
        name="Acme AI",
        description="AI tooling",
    )
    investors = [
        InvestorProfile(
            investor_id="i1",
            name="North Star Ventures",
            description="Updated profile text",
        )
    ]
    real_embedding_service = EmbeddingService(
        EmbeddingSettings(),
        encoder_factory=lambda _: StubEncoder(),
    )
    stale_investors = [
        InvestorProfile(
            investor_id="i1",
            name="North Star Ventures",
            description="Old profile text",
        )
    ]
    index_service = InvestorIndexService(real_embedding_service)
    artifact_dir = _workspace_temp_dir()
    artifact = index_service.build_index(
        investors=stale_investors,
        output_path=artifact_dir / "investor_index.npz",
        source_hash="sample-hash",
    )
    counting_service = CountingEmbeddingService(
        startup_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        investor_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    matching_service = MatchingService(
        embedding_service=counting_service,
        retrieval_service=RetrievalService(RetrievalSettings(default_top_k=1, candidate_pool_size=1)),
        retrieval_settings=RetrievalSettings(default_top_k=1, candidate_pool_size=1),
        scoring_service=ScoringService(ScoringSettings()),
        investor_index_service=index_service,
        investor_index=artifact,
    )

    matches = matching_service.match_startup(startup, investors, top_k=1, candidate_pool_size=1)

    assert len(matches) == 1
    assert counting_service.investor_embed_calls == 1
    _cleanup_temp_dir(artifact_dir)


def _workspace_temp_dir() -> Path:
    temp_dir = Path("tests") / "_tmp" / uuid.uuid4().hex
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def _cleanup_temp_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
