"""2단계 — Faster R-CNN으로 결함 "위치"까지 검출.

1단계 이진분류로는 "이 김에 이물질이 있다"까지만 알 수 있어서, 실제 검사
공정에 쓰려면 위치가 필요하다는 판단으로 객체탐지로 넘어간 단계.
원본 대응 파일: `SeaWeed_Pj/dlddl.py`, `SeaWeed_Pj/seawida.py`,
콜랩 노트북 `Untitled1.ipynb`.

결함 박스가 512x512 이미지에서 평균 11x11px에 불과해 기본 앵커(32~512)로는
잡히지 않았고, 앵커 크기를 (16, 32, 64, 128, 256)으로 내렸다.

실행 예:

    python scripts/stage2_train_faster_rcnn.py \\
        --train-images data/train/train_defected_dataset \\
        --train-csv    data/filtered_seaweed.csv \\
        --val-images   data/validation/validation_defected_dataset \\
        --val-csv      data/val_seaweed.csv \\
        --epochs 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seaweed.config import (  # noqa: E402
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    NUM_CLASSES_MULTICLASS,
)
from seaweed.data import DetectionDataset, collate_detection  # noqa: E402
from seaweed.metrics import DetectionMetrics, accumulate_image  # noqa: E402
from seaweed.models import build_faster_rcnn, get_device  # noqa: E402
from seaweed.transforms import build_eval_transforms, build_train_transforms  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2단계: Faster R-CNN 결함 검출 학습")
    p.add_argument("--train-images", required=True, type=Path)
    p.add_argument("--train-csv", required=True, type=Path)
    p.add_argument("--val-images", required=True, type=Path)
    p.add_argument("--val-csv", required=True, type=Path)
    p.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50"])
    p.add_argument("--epochs", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--augment", action="store_true", help="반전·회전 증강 사용")
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="사전학습 백본을 쓰지 않는다(네트워크가 없는 환경/스모크 테스트용).",
    )
    p.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    p.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    p.add_argument("--output", type=Path, default=Path("outputs/stage2_faster_rcnn.pth"))
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, iou_threshold, score_threshold) -> DetectionMetrics:
    model.eval()
    metrics = DetectionMetrics()
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            accumulate_image(
                metrics,
                pred_boxes=output["boxes"].cpu().numpy(),
                pred_scores=output["scores"].cpu().numpy(),
                pred_labels=output["labels"].cpu().numpy(),
                gt_boxes=target["boxes"].numpy(),
                gt_labels=target["labels"].numpy(),
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
            )
    return metrics


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"[환경] device={device}")

    train_ds = DetectionDataset(
        args.train_images,
        args.train_csv,
        transforms=build_train_transforms() if args.augment else build_eval_transforms(),
    )
    val_ds = DetectionDataset(
        args.val_images, args.val_csv, transforms=build_eval_transforms()
    )
    print(f"[데이터] train={len(train_ds)} / val={len(val_ds)} (증강 포함 개수)")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_detection,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_detection,
        num_workers=args.num_workers,
    )

    model = build_faster_rcnn(
        num_classes=NUM_CLASSES_MULTICLASS,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
    ).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [
                {k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)}
                for t in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            running += losses.item()

        metrics = evaluate(
            model, val_loader, device, args.iou_threshold, args.score_threshold
        )
        print(
            f"Epoch {epoch}/{args.epochs} | loss {running / max(len(train_loader), 1):.4f}\n"
            f"  {metrics.summary()}"
        )

        torch.save(model.state_dict(), args.output)

    print(f"[완료] 모델 저장: {args.output}")


if __name__ == "__main__":
    main()
