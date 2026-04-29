from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class AppSettings(BaseModel):
    """Application-level runtime settings."""

    name: str = "venture-match-engine"
    environment: str = "development"
    debug: bool = False


class DataSettings(BaseModel):
    """File-system locations for raw inputs and local artifacts."""

    data_dir: Path = BASE_DIR / "data"
    artifacts_dir: Path = data_dir / "artifacts"
    startups_csv: str = "examples/startups.sample.csv"
    investors_csv: str = "examples/investors.sample.csv"
    ground_truth_csv: str = "examples/ground_truth.sample.csv"
    investor_index_file: str = "investor_index.npz"

    @property
    def startups_path(self) -> Path:
        return self.data_dir / self.startups_csv

    @property
    def investors_path(self) -> Path:
        return self.data_dir / self.investors_csv

    @property
    def ground_truth_path(self) -> Path:
        return self.data_dir / self.ground_truth_csv

    @property
    def investor_index_path(self) -> Path:
        return self.artifacts_dir / self.investor_index_file


class EmbeddingSettings(BaseModel):
    """Embedding model configuration for semantic retrieval."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = Field(default=32, ge=1)
    normalize_embeddings: bool = True
    cache_dir: Path | None = BASE_DIR / ".cache" / "embeddings"


class RetrievalSettings(BaseModel):
    """Defaults controlling candidate generation before reranking."""

    default_top_k: int = Field(default=10, ge=1)
    candidate_pool_size: int = Field(default=50, ge=1)


class ScoringSettings(BaseModel):
    """Weighted rule-based scoring configuration."""

    semantic_similarity_weight: float = Field(default=0.55, ge=0.0)
    industry_match_weight: float = Field(default=0.20, ge=0.0)
    stage_match_weight: float = Field(default=0.10, ge=0.0)
    geo_match_weight: float = Field(default=0.10, ge=0.0)
    ticket_size_fit_weight: float = Field(default=0.05, ge=0.0)


class Settings(BaseSettings):
    """Top-level application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="VENTURE_MATCH_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    app: AppSettings = AppSettings()
    data: DataSettings = DataSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    scoring: ScoringSettings = ScoringSettings()


@lru_cache
def get_settings() -> Settings:
    return Settings()
