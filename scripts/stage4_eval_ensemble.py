"""4단계 — 결함 유형별 검출기 3개를 앙상블해 검증셋 전체를 평가한다.

원본 대응 파일: `SeaWida_string_1123/dddddddddd.py`(가장 발전된 버전),
`ssshyun_is_the_best.py`(그 이전 버전), `model_ensemble.py`, `counting_iou_low_75.py`.

원본에서 고친 것은 `src/seaweed/ensemble.py` 상단 주석 참고
(검증셋 절반 누락, 시각화용 박스 확대가 IoU를 오염시키던 문제 등).

실행 예:

    python scripts/stage4_eval_ensemble.py \\
        --val-images data/validation/validation_defected_dataset \\
        --val-csv    data/val_seaweed.csv \\
        --checkpoint aq=outputs/stage4/aq/best.pth \\
        --checkpoint st=outputs/stage4/st/best.pth \\
        --checkpoint fl=outputs/stage4/fl/best.pth \\
        --save-ensemble outputs/stage4/ensemble.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seaweed.config import (  # noqa: E402
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    DEFECT_CLASSES,
    LABEL_TO_CLASS,
)
from seaweed.data import DetectionDataset, collate_detection  # noqa: E402
from seaweed.ensemble import DefectEnsemble  # noqa: E402
from seaweed.metrics import DetectionMetrics, calculate_iou  # noqa: E402
from seaweed.models import get_device  # noqa: E402
from seaweed.transforms import build_eval_transforms  # noqa: E402


def parse_checkpoint(value: str) -> tuple[str, Path]:
    """`aq=path/to.pth` 형식을 (클래스, 경로)로 파싱한다."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--checkpoint 는 '클래스=경로' 형식이어야 합니다: {value!r}"
        )
    defect_class, path = value.split("=", 1)
    defect_class = defect_class.strip()
    if defect_class not in DEFECT_CLASSES:
        raise argparse.ArgumentTypeError(
            f"알 수 없는 결함 클래스 {defect_class!r} (사용 가능: {list(DEFECT_CLASSES)})"
        )
    return defect_class, Path(path.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="4단계: 앙상블 평가")
    p.add_argument("--val-images", required=True, type=Path)
    p.add_argument("--val-csv", required=True, type=Path)
    p.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        type=parse_checkpoint,
        metavar="클래스=경로",
        help="결함 클래스별 체크포인트. 예: --checkpoint st=outputs/st/best.pth",
    )
    p.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    p.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    p.add_argument("--save-ensemble", type=Path, default=None)
    p.add_argument("--report", type=Path, default=Path("outputs/stage4/ensemble_report.json"))
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"[환경] device={device}")

    checkpoints = dict(args.checkpoint)
    ensemble = DefectEnsemble(checkpoints, device)

    if args.save_ensemble is not None:
        args.save_ensemble.parent.mkdir(parents=True, exist_ok=True)
        ensemble.save(args.save_ensemble)

    dataset = DetectionDataset(
        args.val_images, args.val_csv, transforms=build_eval_transforms()
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_detection,
        num_workers=args.num_workers,
    )
    print(f"[데이터] 검증 이미지 {len(dataset)}장")

    metrics = DetectionMetrics()
    # 어떤 결함 유형 모델이 몇 번 선택됐고, 그중 몇 번이 정답이었는지.
    selected = Counter()
    selected_correct = Counter()
    iou_fail = Counter()
    no_prediction = 0

    for images, targets in loader:
        # 배치 전체를 순회한다. (원본은 batch_size=2로 만들어놓고 images[0]만 써서
        #  검증셋의 절반을 조용히 건너뛰었다.)
        for image, target in zip(images, targets):
            gt_boxes = target["boxes"].numpy()
            gt_labels = [LABEL_TO_CLASS[int(v)] for v in target["labels"].tolist()]

            box, score, defect_class = ensemble.predict(
                image, score_threshold=args.score_threshold
            )

            if box is None:
                no_prediction += 1
                metrics.false_negatives += len(gt_boxes)
                continue

            selected[defect_class] += 1

            best_iou, best_idx = 0.0, -1
            for idx, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(box, gt_box)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx

            metrics.iou_samples.append(best_iou)

            if best_idx >= 0 and best_iou >= args.iou_threshold:
                metrics.true_positives += 1
                metrics.matched_iou_sum += best_iou
                if gt_labels[best_idx] == defect_class:
                    metrics.class_correct += 1
                    selected_correct[defect_class] += 1
                metrics.false_negatives += len(gt_boxes) - 1
            else:
                metrics.false_positives += 1
                iou_fail[defect_class] += 1
                metrics.false_negatives += len(gt_boxes)

    print("\n=== 앙상블 평가 결과 ===")
    print(metrics.summary())
    print(f"예측 없음(임계값 미달): {no_prediction}장")
    print(f"모델별 선택 횟수: {dict(selected)}")
    print(f"모델별 정답 횟수: {dict(selected_correct)}")
    print(f"모델별 IoU 미달 횟수: {dict(iou_fail)}")

    report = {
        "iou_threshold": args.iou_threshold,
        "score_threshold": args.score_threshold,
        "checkpoints": {k: str(v) for k, v in checkpoints.items()},
        "metrics": metrics.as_dict(),
        "no_prediction": no_prediction,
        "selected_counts": dict(selected),
        "selected_correct": dict(selected_correct),
        "iou_fail_counts": dict(iou_fail),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[완료] 리포트 저장: {args.report}")


if __name__ == "__main__":
    main()
