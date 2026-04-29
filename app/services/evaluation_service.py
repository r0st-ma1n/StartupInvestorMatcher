from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from app.models import MatchResult


class StartupEvaluationMetrics(BaseModel):
    """Per-startup retrieval quality metrics at a fixed cutoff."""

    startup_id: str
    k: int = Field(ge=1)
    precision_at_k: float = Field(ge=0.0, le=1.0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    hit_rate_at_k: float = Field(ge=0.0, le=1.0)
    mrr_at_k: float = Field(ge=0.0, le=1.0)
    predicted_count: int = Field(ge=0)
    relevant_count: int = Field(ge=0)


class EvaluationSummary(BaseModel):
    """Dataset-level summary of ranking quality across startups."""

    k: int = Field(ge=1)
    startup_count: int = Field(ge=0)
    mean_precision_at_k: float = Field(ge=0.0, le=1.0)
    mean_recall_at_k: float = Field(ge=0.0, le=1.0)
    hit_rate_at_k: float = Field(ge=0.0, le=1.0)
    mean_mrr_at_k: float = Field(ge=0.0, le=1.0)
    per_startup: list[StartupEvaluationMetrics] = Field(default_factory=list)


class EvaluationService:
    """Computes simple retrieval and ranking metrics for match results."""

    def evaluate_predictions(
        self,
        predictions: Mapping[str, Sequence[str]],
        ground_truth: Mapping[str, Sequence[str]],
        k: int,
    ) -> EvaluationSummary:
        """Evaluate predicted investor rankings against relevance labels."""

        if k < 1:
            raise ValueError("k must be greater than 0.")

        startup_ids = sorted(set(predictions) | set(ground_truth))
        per_startup: list[StartupEvaluationMetrics] = []
        for startup_id in startup_ids:
            predicted_ids = list(predictions.get(startup_id, []))[:k]
            relevant_ids = set(ground_truth.get(startup_id, []))
            hits = sum(1 for investor_id in predicted_ids if investor_id in relevant_ids)
            reciprocal_rank = 0.0
            for rank, investor_id in enumerate(predicted_ids, start=1):
                if investor_id in relevant_ids:
                    reciprocal_rank = 1.0 / rank
                    break

            precision_at_k = hits / k
            recall_at_k = hits / len(relevant_ids) if relevant_ids else 0.0
            hit_rate_at_k = 1.0 if hits > 0 else 0.0

            per_startup.append(
                StartupEvaluationMetrics(
                    startup_id=startup_id,
                    k=k,
                    precision_at_k=precision_at_k,
                    recall_at_k=recall_at_k,
                    hit_rate_at_k=hit_rate_at_k,
                    mrr_at_k=reciprocal_rank,
                    predicted_count=len(predicted_ids),
                    relevant_count=len(relevant_ids),
                )
            )

        startup_count = len(per_startup)
        if startup_count == 0:
            return EvaluationSummary(
                k=k,
                startup_count=0,
                mean_precision_at_k=0.0,
                mean_recall_at_k=0.0,
                hit_rate_at_k=0.0,
                mean_mrr_at_k=0.0,
                per_startup=[],
            )

        return EvaluationSummary(
            k=k,
            startup_count=startup_count,
            mean_precision_at_k=sum(item.precision_at_k for item in per_startup) / startup_count,
            mean_recall_at_k=sum(item.recall_at_k for item in per_startup) / startup_count,
            hit_rate_at_k=sum(item.hit_rate_at_k for item in per_startup) / startup_count,
            mean_mrr_at_k=sum(item.mrr_at_k for item in per_startup) / startup_count,
            per_startup=per_startup,
        )

    def evaluate_match_results(
        self,
        predictions: Mapping[str, Sequence[MatchResult]],
        ground_truth: Mapping[str, Sequence[str]],
        k: int,
    ) -> EvaluationSummary:
        """Evaluate MatchResult objects directly without pre-extracting IDs."""

        prediction_ids = {
            startup_id: [match.investor_id for match in matches]
            for startup_id, matches in predictions.items()
        }
        return self.evaluate_predictions(prediction_ids, ground_truth, k=k)
