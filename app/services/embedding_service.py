from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EmbeddingSettings
from app.models import InvestorProfile, StartupProfile


class SupportsEncode(Protocol):
    """Protocol for embedders compatible with sentence-transformers."""

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray: ...


class EmbeddingService:
    """Generates dense vector representations for startup and investor profiles."""

    def __init__(
        self,
        settings: EmbeddingSettings,
        encoder_factory: Callable[[EmbeddingSettings], SupportsEncode] | None = None,
    ) -> None:
        self._settings = settings
        self._encoder_factory = encoder_factory or self._build_default_encoder
        self._encoder: SupportsEncode | None = None

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed raw texts into a dense NumPy matrix."""

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        cleaned_texts = [self._clean_text(text) for text in texts]
        embeddings = self._get_encoder().encode(
            cleaned_texts,
            batch_size=self._settings.batch_size,
            normalize_embeddings=self._settings.normalize_embeddings,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_startups(self, startups: Sequence[StartupProfile]) -> np.ndarray:
        """Embed startup profiles using a deterministic text representation."""

        return self.embed_texts([self.format_startup_text(startup) for startup in startups])

    def embed_investors(self, investors: Sequence[InvestorProfile]) -> np.ndarray:
        """Embed investor profiles using a deterministic text representation."""

        return self.embed_texts([self.format_investor_text(investor) for investor in investors])

    @staticmethod
    def format_startup_text(startup: StartupProfile) -> str:
        """Create the text used for semantic retrieval of startup profiles."""

        return _join_sections(
            [
                startup.name,
                startup.description,
                _prefixed_section("Industries", startup.industries),
                _prefixed_section("Stage", startup.stage),
                _prefixed_section("Country", startup.country),
                _prefixed_section("Region", startup.region),
            ]
        )

    @staticmethod
    def format_investor_text(investor: InvestorProfile) -> str:
        """Create the text used for semantic retrieval of investor profiles."""

        return _join_sections(
            [
                investor.name,
                investor.description,
                _prefixed_section("Industries", investor.industries),
                _prefixed_section("Preferred stages", investor.preferred_stages),
                _prefixed_section("Countries", investor.countries),
                _prefixed_section("Regions", investor.regions),
            ]
        )

    def _get_encoder(self) -> SupportsEncode:
        if self._encoder is None:
            self._encoder = self._encoder_factory(self._settings)
        return self._encoder

    @staticmethod
    def _build_default_encoder(settings: EmbeddingSettings) -> SupportsEncode:
        return SentenceTransformer(
            settings.model_name,
            cache_folder=str(settings.cache_dir) if settings.cache_dir else None,
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(text.split())


def _prefixed_section(prefix: str, value: str | Sequence[str] | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        return f"{prefix}: {text}" if text else ""

    items = [item.strip() for item in value if item and item.strip()]
    if not items:
        return ""
    return f"{prefix}: {', '.join(items)}"


def _join_sections(parts: Sequence[str]) -> str:
    return "\n".join(part for part in parts if part)
