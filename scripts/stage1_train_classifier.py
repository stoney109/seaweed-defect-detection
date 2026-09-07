"""1단계 — 김 이미지 결함/무결함 이진분류 (ResNet).

가장 먼저 시도한 접근. "이 이미지에 이물질이 있는가?"만 판단하고 위치는 찾지
않는다. 원본에서는 이 단계 스크립트가 `SeaWeed_Pj/` 안에 8개 이상 복사되어
있었고(`ResNet.py`, `add_preprocess.py`, `fliped_image.py`, `parameter_fixed.py`,
`dkd.py`, `tjgusdlrk_model.py` 등) 서로 학습률·배치크기·클래스가중치·증강만
달랐다. 그 변형들을 전부 CLI 인자로 옮겼다.

실행 예:

    python scripts/stage1_train_classifier.py \\
        --train-defected data/train/train_defected_dataset \\
        --train-pure     data/train/train_pure_dataset \\
        --val-defected   data/validation/validation_defected_dataset \\
        --val-pure       data/validation/validation_pure_dataset \\
        --max-pure 3000 --epochs 10 --pos-weight 2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seaweed.data import ClassificationDataset  # noqa: E402
from seaweed.metrics import binary_classification_metrics  # noqa: E402
from seaweed.models import build_classifier, get_device  # noqa: E402
from seaweed.transforms import build_classification_transform  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1단계: 결함/무결함 이진분류 학습")
    p.add_argument("--train-defected", required=True, type=Path)
    p.add_argument("--train-pure", required=True, type=Path)
    p.add_argument("--val-defected", required=True, type=Path)
    p.add_argument("--val-pure", required=True, type=Path)
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument(
        "--pos-weight",
        type=float,
        default=2.0,
        help="결함 클래스 가중치. 무결함이 압도적으로 많아 필요하다.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="시그모이드 출력 이진화 임계값.",
    )
    p.add_argument(
        "--max-pure",
        type=int,
        default=3000,
        help=(
            "무결함 이미지 사용 개수 상한. 원본 데이터는 결함 3,000장 대 "
            "무결함 255,000장이라 상한 없이 쓰면 학습이 무너진다."
        ),
    )
    p.add_argument("--augment", action="store_true", help="학습셋에 반전 증강 적용")
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="ImageNet 사전학습 가중치를 쓰지 않는다(네트워크가 없는 환경/스모크 테스트용).",
    )
    p.add_argument("--output", type=Path, default=Path("outputs/stage1_classifier.pth"))
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, threshold: float) -> dict:
    model.eval()
    all_labels, all_preds = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).int().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(np.asarray(labels).astype(int))
    if not all_labels:
        return {}
    return binary_classification_metrics(
        np.concatenate(all_labels), np.concatenate(all_preds)
    )


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"[환경] device={device}")

    train_ds = ClassificationDataset(
        args.train_defected,
        args.train_pure,
        transform=build_classification_transform(train=args.augment),
        max_pure=args.max_pure,
    )
    val_ds = ClassificationDataset(
        args.val_defected,
        args.val_pure,
        transform=build_classification_transform(train=False),
        max_pure=args.max_pure,
    )
    print(f"[데이터] train={train_ds.class_balance} / val={val_ds.class_balance}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_classifier(args.backbone, pretrained=not args.no_pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], device=device)
    )
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.optimizer == "adam"
        else torch.optim.SGD(model.parameters(), lr=args.lr)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()

        metrics = evaluate(model, val_loader, device, args.threshold)
        print(
            f"Epoch {epoch}/{args.epochs} | loss {running / max(len(train_loader), 1):.4f} "
            f"| val {metrics}"
        )

        if metrics.get("f1", 0.0) >= best_f1:
            best_f1 = metrics.get("f1", 0.0)
            torch.save(model.state_dict(), args.output)
            print(f"  best 갱신 -> 저장: {args.output}")

    print(f"[완료] best val F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
