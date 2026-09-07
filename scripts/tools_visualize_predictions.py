"""예측 결과 시각화 — 정답 박스(파랑)와 예측 박스(빨강)를 겹쳐 그린다.

결함이 11px 남짓이라 512px 이미지에 그대로 그리면 잘 보이지 않는다. 그래서
**시각화할 때만** 박스를 최소 크기까지 키운다.

원본 `dddddddddd.py`는 이 확대를 예측 함수(`get_predictions`) 안에서 해버려서,
확대된 박스가 그대로 IoU 계산에 들어가 지표를 왜곡했다. 여기서는 시각화
전용으로 분리했다 — 평가 경로(`stage4_eval_ensemble.py`)는 원본 크기를 쓴다.

원본 대응 파일: `SeaWida/visual_False_iou_bound.py`,
`SeaWida_string_1123/visual_bessssst_iou_hist.py` 등.

실행 예:

    python scripts/tools_visualize_predictions.py \\
        --val-images data/validation/validation_defected_dataset \\
        --val-csv    data/val_seaweed.csv \\
        --checkpoint aq=outputs/stage4/aq/best.pth \\
        --checkpoint st=outputs/stage4/st/best.pth \\
        --checkpoint fl=outputs/stage4/fl/best.pth \\
        --num-images 12 --only-failures --output-dir outputs/vis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seaweed.config import (  # noqa: E402
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    DEFECT_CLASSES,
    LABEL_TO_CLASS,
)
from seaweed.data import DetectionDataset, collate_detection  # noqa: E402
from seaweed.ensemble import DefectEnsemble  # noqa: E402
from seaweed.metrics import calculate_iou  # noqa: E402
from seaweed.models import get_device  # noqa: E402
from seaweed.transforms import build_eval_transforms  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

MIN_DRAW_SIZE = 24  # 시각화 전용 최소 박스 크기(px)


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--checkpoint 는 '클래스=경로' 형식이어야 합니다.")
    defect_class, path = value.split("=", 1)
    if defect_class.strip() not in DEFECT_CLASSES:
        raise argparse.ArgumentTypeError(f"알 수 없는 결함 클래스: {defect_class!r}")
    return defect_class.strip(), Path(path.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="앙상블 예측 시각화")
    p.add_argument("--val-images", required=True, type=Path)
    p.add_argument("--val-csv", required=True, type=Path)
    p.add_argument("--checkpoint", action="append", required=True, type=parse_checkpoint)
    p.add_argument("--num-images", type=int, default=10)
    p.add_argument(
        "--only-failures",
        action="store_true",
        help="IoU 임계값을 넘지 못한 사례만 저장 (오답 분석용)",
    )
    p.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    p.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/visualizations"))
    return p.parse_args()


def enlarge_for_drawing(box, min_size: int = MIN_DRAW_SIZE):
    """시각화 전용 — 너무 작은 박스를 눈에 보이게 키운다(평가에는 쓰지 않는다)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(min_size, x2 - x1, y2 - y1) / 2
    return [cx - half, cy - half, cx + half, cy + half]


def main() -> None:
    args = parse_args()
    device = get_device()

    ensemble = DefectEnsemble(dict(args.checkpoint), device)
    dataset = DetectionDataset(
        args.val_images, args.val_csv, transforms=build_eval_transforms()
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_detection
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for images, targets in loader:
        if saved >= args.num_images:
            break

        image_tensor, target = images[0], targets[0]
        gt_boxes = target["boxes"].numpy()
        gt_labels = [LABEL_TO_CLASS[int(v)] for v in target["labels"].tolist()]
        img_name = target["image_name"]

        box, score, defect_class = ensemble.predict(
            image_tensor, score_threshold=args.score_threshold
        )

        best_iou = 0.0
        if box is not None:
            best_iou = max((calculate_iou(box, g) for g in gt_boxes), default=0.0)

        is_failure = box is None or best_iou < args.iou_threshold
        if args.only_failures and not is_failure:
            continue

        # 원본 이미지를 다시 열어 그린다(정규화된 텐서를 되돌리지 않기 위해).
        canvas = Image.open(args.val_images / img_name).convert("RGB")
        draw = ImageDraw.Draw(canvas)

        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            draw.rectangle(enlarge_for_drawing(gt_box), outline="blue", width=3)
            draw.text((gt_box[0], max(gt_box[1] - 14, 0)), f"GT:{gt_label}", fill="blue")

        if box is not None:
            draw.rectangle(enlarge_for_drawing(box), outline="red", width=3)
            draw.text(
                (box[0], min(box[3] + 4, canvas.height - 12)),
                f"{defect_class} {score:.2f} IoU {best_iou:.2f}",
                fill="red",
            )

        prefix = "fail" if is_failure else "ok"
        out = args.output_dir / f"{prefix}_{Path(img_name).stem}.png"
        canvas.save(out)
        saved += 1

    print(f"[완료] {saved}장 저장 -> {args.output_dir}")
    print("파랑=정답, 빨강=예측. 작은 박스는 보기 좋게 확대해 그린 것이며 지표와 무관하다.")


if __name__ == "__main__":
    main()
