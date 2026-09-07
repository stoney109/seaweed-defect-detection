"""검출/분류 평가 지표.

## 원본 코드의 평가 함수 두 종류

원본에는 성격이 다른 평가 함수가 두 갈래로 존재했다.

- `SeaWida_1116/*.py`의 `evaluate_model_with_class_verification()`
  → 예측 하나가 여러 GT와 겹치면 IoU를 **중복 누적**하고,
    `false_positives += len(pred_boxes) - len(matched_gt)`처럼 신뢰도 필터링
    이전의 전체 예측 수로 FP를 계산해 precision이 실제보다 낮게 나온다.
- `SeaWida_string/Retina_st_model.py`의 `evaluate_model()`
  → 예측마다 IoU가 가장 큰 미매칭 GT를 고르는 그리디 매칭. 이쪽이 맞다.

여기서는 후자를 채택하고, 전자에만 있던 "클래스까지 맞았는가" 집계를 합쳤다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def calculate_iou(box_a, box_b) -> float:
    """두 박스(x_min, y_min, x_max, y_max)의 IoU."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


@dataclass
class DetectionMetrics:
    """검출 성능 집계 결과."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    class_correct: int = 0
    matched_iou_sum: float = 0.0
    iou_samples: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def average_iou(self) -> float:
        """TP로 인정된 매칭들의 평균 IoU."""
        return self.matched_iou_sum / self.true_positives if self.true_positives else 0.0

    @property
    def class_accuracy(self) -> float:
        """위치를 맞춘(TP) 예측 중 클래스까지 맞은 비율."""
        return self.class_correct / self.true_positives if self.true_positives else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "average_iou": round(self.average_iou, 4),
            "class_accuracy": round(self.class_accuracy, 4),
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
        }

    def summary(self) -> str:
        d = self.as_dict()
        return (
            f"Precision {d['precision']:.4f} | Recall {d['recall']:.4f} | "
            f"F1 {d['f1']:.4f} | Avg IoU {d['average_iou']:.4f} | "
            f"Class Acc {d['class_accuracy']:.4f} | "
            f"TP {d['tp']} FP {d['fp']} FN {d['fn']}"
        )


def accumulate_image(
    metrics: DetectionMetrics,
    pred_boxes,
    pred_scores,
    pred_labels,
    gt_boxes,
    gt_labels,
    iou_threshold: float,
    score_threshold: float,
) -> None:
    """이미지 한 장의 예측을 그리디 매칭해 지표에 누적한다.

    규칙:
      - 신뢰도가 `score_threshold` 미만인 예측은 아예 무시한다(FP로도 세지 않는다).
      - 남은 예측을 신뢰도 높은 순으로 훑으며, 아직 매칭되지 않은 GT 중 IoU가
        가장 큰 것을 고른다. 그 IoU가 임계값 이상이면 TP, 아니면 FP.
      - 끝까지 매칭되지 않은 GT는 FN.
    """
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=float).reshape(-1)
    pred_labels = np.asarray(pred_labels).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    gt_labels = np.asarray(gt_labels).reshape(-1)

    keep = pred_scores >= score_threshold
    pred_boxes, pred_scores, pred_labels = (
        pred_boxes[keep],
        pred_scores[keep],
        pred_labels[keep],
    )

    order = np.argsort(-pred_scores)  # 신뢰도 내림차순
    matched_gt: set[int] = set()

    for pred_idx in order:
        best_iou, best_gt = 0.0, -1
        for gt_idx in range(len(gt_boxes)):
            if gt_idx in matched_gt:
                continue
            iou = calculate_iou(pred_boxes[pred_idx], gt_boxes[gt_idx])
            if iou > best_iou:
                best_iou, best_gt = iou, gt_idx

        metrics.iou_samples.append(best_iou)

        if best_gt >= 0 and best_iou >= iou_threshold:
            metrics.true_positives += 1
            metrics.matched_iou_sum += best_iou
            matched_gt.add(best_gt)
            if len(pred_labels) and len(gt_labels):
                if pred_labels[pred_idx] == gt_labels[best_gt]:
                    metrics.class_correct += 1
        else:
            metrics.false_positives += 1

    metrics.false_negatives += len(gt_boxes) - len(matched_gt)


def binary_classification_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    """1단계 이진분류용 accuracy / precision / recall / f1."""
    labels = np.asarray(labels).astype(int).reshape(-1)
    predictions = np.asarray(predictions).astype(int).reshape(-1)

    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


__all__ = [
    "calculate_iou",
    "DetectionMetrics",
    "accumulate_image",
    "binary_classification_metrics",
]
