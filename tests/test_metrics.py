"""IoU 계산과 검출 지표 집계 검증."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seaweed.metrics import (  # noqa: E402
    DetectionMetrics,
    accumulate_image,
    binary_classification_metrics,
    calculate_iou,
)


def test_iou_identical_boxes() -> None:
    box = [10, 10, 20, 20]
    assert calculate_iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes() -> None:
    assert calculate_iou([0, 0, 10, 10], [50, 50, 60, 60]) == pytest.approx(0.0)


def test_iou_half_overlap() -> None:
    # 10x10 두 박스가 x축으로 절반 겹침 -> 교집합 50, 합집합 150
    assert calculate_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_iou_zero_area_box() -> None:
    assert calculate_iou([5, 5, 5, 5], [0, 0, 10, 10]) == pytest.approx(0.0)


def test_perfect_prediction() -> None:
    m = DetectionMetrics()
    accumulate_image(
        m,
        pred_boxes=[[10, 10, 20, 20]],
        pred_scores=[0.9],
        pred_labels=[1],
        gt_boxes=[[10, 10, 20, 20]],
        gt_labels=[1],
        iou_threshold=0.75,
        score_threshold=0.5,
    )
    assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 0, 0)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)
    assert m.class_accuracy == pytest.approx(1.0)


def test_low_confidence_prediction_is_ignored_not_counted_as_fp() -> None:
    """신뢰도 미달 예측은 FP로 세지 않는다.

    원본 평가 함수는 신뢰도 필터링 이전의 예측 개수로 FP를 계산해서
    precision이 실제보다 낮게 나왔다.
    """
    m = DetectionMetrics()
    accumulate_image(
        m,
        pred_boxes=[[10, 10, 20, 20]],
        pred_scores=[0.1],  # 임계값 미달
        pred_labels=[1],
        gt_boxes=[[10, 10, 20, 20]],
        gt_labels=[1],
        iou_threshold=0.75,
        score_threshold=0.5,
    )
    assert m.false_positives == 0
    assert m.false_negatives == 1  # 정답을 못 잡았으므로 FN
    assert m.true_positives == 0


def test_one_prediction_matches_only_one_gt() -> None:
    """예측 하나가 GT 두 개를 동시에 맞춘 것으로 세지 않는다."""
    m = DetectionMetrics()
    accumulate_image(
        m,
        pred_boxes=[[10, 10, 20, 20]],
        pred_scores=[0.9],
        pred_labels=[1],
        gt_boxes=[[10, 10, 20, 20], [10, 10, 20, 20]],
        gt_labels=[1, 1],
        iou_threshold=0.75,
        score_threshold=0.5,
    )
    assert m.true_positives == 1
    assert m.false_negatives == 1


def test_wrong_class_still_counts_as_localization_tp() -> None:
    """위치는 맞고 클래스만 틀린 경우 — TP지만 class_accuracy에서 빠진다."""
    m = DetectionMetrics()
    accumulate_image(
        m,
        pred_boxes=[[10, 10, 20, 20]],
        pred_scores=[0.9],
        pred_labels=[2],
        gt_boxes=[[10, 10, 20, 20]],
        gt_labels=[1],
        iou_threshold=0.75,
        score_threshold=0.5,
    )
    assert m.true_positives == 1
    assert m.class_correct == 0
    assert m.class_accuracy == pytest.approx(0.0)


def test_greedy_matching_prefers_best_iou() -> None:
    """겹치는 GT가 여럿이면 IoU가 가장 큰 쪽에 매칭된다."""
    m = DetectionMetrics()
    accumulate_image(
        m,
        pred_boxes=[[10, 10, 20, 20]],
        pred_scores=[0.9],
        pred_labels=[1],
        gt_boxes=[[0, 0, 10, 10], [10, 10, 20, 20]],
        gt_labels=[2, 1],
        iou_threshold=0.75,
        score_threshold=0.5,
    )
    assert m.true_positives == 1
    assert m.class_correct == 1  # 라벨 1인 두 번째 GT에 매칭됐다는 뜻


def test_empty_metrics_do_not_divide_by_zero() -> None:
    m = DetectionMetrics()
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert m.average_iou == 0.0
    assert m.class_accuracy == 0.0


def test_binary_classification_metrics() -> None:
    result = binary_classification_metrics(
        labels=[1, 1, 0, 0, 1], predictions=[1, 0, 0, 1, 1]
    )
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tn"] == 1
    assert result["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["accuracy"] == pytest.approx(0.6)
