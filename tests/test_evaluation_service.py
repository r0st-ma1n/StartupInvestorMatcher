from __future__ import annotations

import pytest

from app.models import Candidate, InvestorProfile, MatchResult, ScoreBreakdown
from app.services import EvaluationService


def test_evaluate_predictions_computes_mean_metrics() -> None:
    service = EvaluationService()

    summary = service.evaluate_predictions(
        predictions={
            "s1": ["i1", "i2"],
            "s2": ["i5", "i6"],
        },
        ground_truth={
            "s1": ["i2", "i3"],
            "s2": ["i9"],
        },
        k=2,
    )

    assert summary.startup_count == 2
    assert summary.mean_precision_at_k == pytest.approx(0.25)
    assert summary.mean_recall_at_k == pytest.approx(0.25)
    assert summary.hit_rate_at_k == pytest.approx(0.5)
    assert summary.mean_mrr_at_k == pytest.approx(0.25)


def test_evaluate_match_results_extracts_investor_ids() -> None:
    service = EvaluationService()
    predictions = {
        "s1": [
            MatchResult(
                startup_id="s1",
                investor_id="i1",
                investor_name="Alpha",
                rank=1,
                score=0.9,
                candidate=Candidate(
                    investor=InvestorProfile(
                        investor_id="i1",
                        name="Alpha",
                        description="Fund",
                    ),
                    semantic_similarity=0.9,
                    retrieval_rank=1,
                ),
                score_breakdown=ScoreBreakdown(
                    semantic_similarity=0.9,
                    industry_match=1.0,
                    stage_match=1.0,
                    geo_match=1.0,
                    ticket_size_fit=1.0,
                    weighted_score=0.95,
                ),
            )
        ]
    }

    summary = service.evaluate_match_results(
        predictions=predictions,
        ground_truth={"s1": ["i1", "i2"]},
        k=1,
    )

    assert summary.mean_precision_at_k == pytest.approx(1.0)
    assert summary.mean_recall_at_k == pytest.approx(0.5)
    assert summary.hit_rate_at_k == pytest.approx(1.0)
    assert summary.mean_mrr_at_k == pytest.approx(1.0)


def test_evaluate_predictions_rejects_invalid_k() -> None:
    service = EvaluationService()

    with pytest.raises(ValueError):
        service.evaluate_predictions(predictions={}, ground_truth={}, k=0)


def test_evaluate_predictions_penalizes_short_ranked_lists_at_k() -> None:
    service = EvaluationService()

    summary = service.evaluate_predictions(
        predictions={"s1": ["i1"]},
        ground_truth={"s1": ["i1", "i2"]},
        k=3,
    )

    assert summary.mean_precision_at_k == pytest.approx(1 / 3)
    assert summary.mean_recall_at_k == pytest.approx(0.5)
    assert summary.mean_mrr_at_k == pytest.approx(1.0)
