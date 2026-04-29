from __future__ import annotations

from app.config import get_settings
from app.services import CatalogService, EmbeddingService, InvestorIndexService


def main() -> None:
    settings = get_settings()
    catalog_service = CatalogService(
        startups_path=settings.data.startups_path,
        investors_path=settings.data.investors_path,
    )
    investors = catalog_service.list_investors()

    embedding_service = EmbeddingService(settings.embeddings)
    index_service = InvestorIndexService(embedding_service)
    source_hash = index_service.compute_source_hash(settings.data.investors_path)
    artifact = index_service.build_index(
        investors=investors,
        output_path=settings.data.investor_index_path,
        source_hash=source_hash,
    )
    print(
        f"Built investor index at {settings.data.investor_index_path} "
        f"for {len(artifact.investor_ids)} investors."
    )


if __name__ == "__main__":
    main()
