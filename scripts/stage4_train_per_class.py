"""4단계 — 결함 유형별 단일 클래스 검출기 학습 (앙상블의 구성원).

3단계까지는 결함 3종(aq/st/fl)을 한 모델이 동시에 분류·검출했는데, 클래스별
데이터 수가 불균형하고(st 1289 / aq 1088 / fl 623) 결함 모양도 서로 달라서
한 모델이 세 유형을 모두 잘 잡지 못했다. 그래서 유형마다 "배경 vs 해당 결함"
2클래스 검출기를 따로 학습시키는 쪽으로 전환한 것이 마지막 단계다.

원본 대응 파일: `SeaWida_string/Retina_st_model.py`,
`SeaWida_string/Retina_st_addtional_train.py`,
`SeaWida_string_1123/` 의 같은 파일들.
원본은 학습할 클래스를 바꾸려면 소스 안의 `== "fl"` 문자열을 직접 고쳐야 했다
(그래서 파일명은 `st`인데 내용은 `fl`을 거르는 상태로 남아 있었다).
여기서는 `--defect-class` 인자로 받는다.

실행 예 (세 유형을 각각 학습):

    for C in aq st fl; do
      python scripts/stage4_train_per_class.py --defect-class $C \\
        --train-images data/train/train_defected_dataset \\
        --train-csv    data/filtered_seaweed.csv \\
        --val-images   data/validation/validation_defected_dataset \\
        --val-csv      data/val_seaweed.csv \\
        --augment --epochs 30 --output-dir outputs/stage4/$C
    done
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seaweed.config import (  # noqa: E402
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    DEFECT_CLASSES,
    NUM_CLASSES_PER_CLASS_DETECTOR,
)
from seaweed.data import PerClassDetectionDataset, collate_detection  # noqa: E402
from seaweed.metrics import DetectionMetrics, accumulate_image  # noqa: E402
from seaweed.models import build_retinanet, get_device, load_checkpoint  # noqa: E402
from seaweed.transforms import build_eval_transforms, build_train_transforms  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="4단계: 결함 유형별 단일 클래스 검출기 학습")
    p.add_argument("--defect-class", required=True, choices=list(DEFECT_CLASSES))
    p.add_argument("--train-images", required=True, type=Path)
    p.add_argument("--train-csv", required=True, type=Path)
    p.add_argument("--val-images", required=True, type=Path)
    p.add_argument("--val-csv", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--augment", action="store_true", help="반전·회전 증강 6종 사용")
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="COCO 사전학습 가중치를 쓰지 않는다(네트워크가 없는 환경/스모크 테스트용).",
    )
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--start-epoch", type=int, default=1)
    p.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    p.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    p.add_argument("--output-dir", type=Path, default=None)
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
    output_dir = args.output_dir or Path(f"outputs/stage4/{args.defect_class}")
    device = get_device()
    print(f"[환경] device={device} / 대상 결함 클래스={args.defect_class}")

    train_ds = PerClassDetectionDataset(
        args.train_images,
        args.train_csv,
        defect_class=args.defect_class,
        transforms=build_train_transforms() if args.augment else build_eval_transforms(),
    )
    val_ds = PerClassDetectionDataset(
        args.val_images,
        args.val_csv,
        defect_class=args.defect_class,
        transforms=build_eval_transforms(),
    )
    print(f"[데이터] train={len(train_ds)} / val={len(val_ds)}")

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
        num_classes=NUM_CLASSES_PER_CLASS_DETECTOR, pretrained=not args.no_pretrained
    )
    if args.resume is not None:
        model = load_checkpoint(model, args.resume, device)
    else:
        model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
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

        train_loss = running / max(len(train_loader), 1)
        metrics = evaluate(
            model, val_loader, device, args.iou_threshold, args.score_threshold
        )
        print(f"Epoch #{epoch} Train Loss: {train_loss:.4f}\n  {metrics.summary()}")

        checkpoint = output_dir / f"retinanet_{args.defect_class}_epoch_{epoch:02d}.pth"
        torch.save(model.state_dict(), checkpoint)

        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), **metrics.as_dict()})
        # 원본에서는 에폭별 성능을 손으로 txt에 옮겨 적었다(st_model_validation.txt).
        # 여기서는 매 에폭 JSON으로 남겨 나중에 표/그래프로 바로 쓸 수 있게 한다.
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    best = max(history, key=lambda h: h["f1"]) if history else None
    if best:
        print(f"[완료] 최고 F1 = {best['f1']} (epoch {best['epoch']})")


if __name__ == "__main__":
    main()
