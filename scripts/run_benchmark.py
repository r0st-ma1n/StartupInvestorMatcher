from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from app.config import get_settings
from app.services import (
    CatalogService,
    EmbeddingService,
    EvaluationService,
    MatchingService,
    RetrievalService,
    ScoringService,
)


def _load_ground_truth(path: str | Path) -> dict[str, list[str]]:
    ground_truth: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            startup_id = row["startup_id"].strip()
            investor_id = row["investor_id"].strip()
            ground_truth.setdefault(startup_id, []).append(investor_id)
    return ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval and reranking benchmark on sample data.")
    parser.add_argument("--k", type=int, default=2, help="Cutoff for ranking metrics.")
    parser.add_argument("--startups-path", type=Path, default=None)
    parser.add_argument("--investors-path", type=Path, default=None)
    parser.add_argument("--ground-truth-path", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    startups_path = args.startups_path or settings.data.startups_path
    investors_path = args.investors_path or settings.data.investors_path
    ground_truth_path = args.ground_truth_path or settings.data.ground_truth_path
    catalog_service = CatalogService(
        startups_path=startups_path,
        investors_path=investors_path,
    )
    matching_service = MatchingService(
        embedding_service=EmbeddingService(settings.embeddings),
        retrieval_service=RetrievalService(settings.retrieval),
        retrieval_settings=settings.retrieval,
        scoring_service=ScoringService(settings.scoring),
    )
    evaluation_service = EvaluationService()

    startups = catalog_service.list_startups()
    investors = catalog_service.list_investors()
    ground_truth = _load_ground_truth(ground_truth_path)

    semantic_predictions = {
        startup.startup_id: [
            candidate.investor.investor_id
            for candidate in matching_service.match_startup_semantic(startup, investors, top_k=args.k)
        ]
        for startup in startups
    }
    reranked_predictions = {
        startup.startup_id: [
            match.investor_id
            for match in matching_service.match_startup(
                startup,
                investors,
                top_k=args.k,
                candidate_pool_size=max(args.k, settings.retrieval.candidate_pool_size),
            )
        ]
        for startup in startups
    }

    semantic_metrics = evaluation_service.evaluate_predictions(
        predictions=semantic_predictions,
        ground_truth=ground_truth,
        k=args.k,
    )
    reranked_metrics = evaluation_service.evaluate_predictions(
        predictions=reranked_predictions,
        ground_truth=ground_truth,
        k=args.k,
    )

    print(
        json.dumps(
            {
                "k": args.k,
                "semantic_only": semantic_metrics.model_dump(),
                "retrieval_plus_reranking": reranked_metrics.model_dump(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
