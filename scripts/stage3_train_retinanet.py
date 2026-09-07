"""3단계 — RetinaNet + 데이터 증강 + 체크포인트 이어학습.

Faster R-CNN이 작은 결함을 잘 못 잡아서, 작은 객체에 강한 1-stage 검출기인
RetinaNet(ResNet50-FPN, COCO 사전학습)으로 갈아탄 단계.
원본 대응 파일: `SeaWida_1116/RetinaNet.py` → `RetinaNet_csv_ver.py` →
`RetinaNet_saveRepeat.py` → `Retina_data_augmentation.py` →
`Retina_new_additional.py` → `Retina_addtional.py` (6단계에 걸친 복붙 개선),
콜랩 노트북 `Untitled6.ipynb`.

이 단계에서 붙인 기능들:
  - 에폭마다 체크포인트 저장 (`--save-every-epoch`)
  - 저장된 체크포인트에서 이어학습 (`--resume`)
  - 반전·회전 증강 (박스 좌표 동기 변환 — 원본의 버그를 고친 부분)
  - 박스 정규화 실험 (`--normalize-box-size`, 원본
    `SeaWida/retina_boundingbox_normalization.py`)

실행 예:

    python scripts/stage3_train_retinanet.py \\
        --train-images data/train/train_defected_dataset \\
        --train-csv    data/filtered_seaweed.csv \\
        --val-images   data/validation/validation_defected_dataset \\
        --val-csv      data/val_seaweed.csv \\
        --augment --epochs 20 --output-dir outputs/stage3
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
    NORMALIZED_BOX_SIZE,
    NUM_CLASSES_MULTICLASS,
)
from seaweed.data import DetectionDataset, collate_detection  # noqa: E402
from seaweed.metrics import DetectionMetrics, accumulate_image  # noqa: E402
from seaweed.models import build_retinanet, get_device, load_checkpoint  # noqa: E402
from seaweed.transforms import build_eval_transforms, build_train_transforms  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3단계: RetinaNet 결함 검출 학습")
    p.add_argument("--train-images", required=True, type=Path)
    p.add_argument("--train-csv", required=True, type=Path)
    p.add_argument("--val-images", required=True, type=Path)
    p.add_argument("--val-csv", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--augment", action="store_true", help="반전·회전 증강 6종 사용")
    p.add_argument(
        "--normalize-box-size",
        type=int,
        nargs="?",
        const=NORMALIZED_BOX_SIZE,
        default=None,
        help=(
            "정답 박스를 중심 유지 정사각형으로 치환하는 실험 "
            f"(값 생략 시 {NORMALIZED_BOX_SIZE}px)."
        ),
    )
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="COCO 사전학습 가중치를 쓰지 않는다(네트워크가 없는 환경/스모크 테스트용).",
    )
    p.add_argument("--resume", type=Path, default=None, help="이어학습할 체크포인트")
    p.add_argument("--start-epoch", type=int, default=1, help="이어학습 시작 에폭 번호")
    p.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    p.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/stage3"))
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
        normalize_box_size=args.normalize_box_size,
    )
    val_ds = DetectionDataset(
        args.val_images,
        args.val_csv,
        transforms=build_eval_transforms(),
        normalize_box_size=args.normalize_box_size,
    )
    print(
        f"[데이터] train={len(train_ds)} (증강 {len(train_ds.transforms)}종) / "
        f"val={len(val_ds)}"
    )
    if args.normalize_box_size:
        print(f"[실험] 정답 박스를 {args.normalize_box_size}px 정사각형으로 정규화")

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

    model = build_retinanet(
        num_classes=NUM_CLASSES_MULTICLASS, pretrained=not args.no_pretrained
    )
    if args.resume is not None:
        model = load_checkpoint(model, args.resume, device)
    else:
        model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_epoch = args.start_epoch + args.epochs - 1

    for epoch in range(args.start_epoch, last_epoch + 1):
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
            f"Epoch #{epoch} | Train Loss {running / max(len(train_loader), 1):.4f}\n"
            f"  {metrics.summary()}"
        )

        checkpoint = args.output_dir / f"retinanet_epoch_{epoch:02d}.pth"
        torch.save(model.state_dict(), checkpoint)
        print(f"  체크포인트 저장: {checkpoint}")

    print("[완료] 3단계 학습 종료")


if __name__ == "__main__":
    main()
