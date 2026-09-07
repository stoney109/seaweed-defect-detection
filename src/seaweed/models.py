"""모델 생성 함수 모음 (1~4단계에서 쓴 세 가지 아키텍처).

원본에서는 모델 정의가 학습 스크립트마다 인라인으로 들어가 있었고, 백본 종류
(resnet18/resnet50)와 사전학습 가중치 사용 여부가 파일마다 달랐다.
여기서는 팩토리 함수 세 개로 정리하고 기본값을 통일했다.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from torchvision.models.detection import (
    FasterRCNN,
    RetinaNet_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn,
)
from torchvision.models.detection.retinanet import (
    RetinaNetClassificationHead,
    RetinaNetRegressionHead,
)
from torchvision.models.detection.rpn import AnchorGenerator


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return device


# ---------------------------------------------------------------------------
# 1단계: ResNet 이진분류기
# ---------------------------------------------------------------------------
def build_classifier(backbone: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """결함/무결함 이진분류용 ResNet.

    출력 노드는 1개이며 손실은 `BCEWithLogitsLoss`를 쓴다(시그모이드는 손실
    함수 안에 포함되므로 모델에는 넣지 않는다).
    """
    if backbone == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
    elif backbone == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.resnet50(weights=weights)
    else:
        raise ValueError(f"지원하지 않는 backbone: {backbone!r} (resnet18|resnet50)")

    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


# ---------------------------------------------------------------------------
# 2단계: Faster R-CNN
# ---------------------------------------------------------------------------
def build_faster_rcnn(
    num_classes: int,
    backbone: str = "resnet50",
    anchor_sizes: tuple[int, ...] = (16, 32, 64, 128, 256),
    pretrained: bool = True,
) -> FasterRCNN:
    """ResNet 백본 위에 얹은 Faster R-CNN.

    결함 박스가 평균 11px로 매우 작아서 앵커 크기를 기본값보다 작은 쪽으로
    내렸다. 원본에서도 `(32, 64, 128, 256, 512)` -> `(16, 32, 64, 128, 256)`으로
    한 번 내린 흔적이 있다.
    """
    if backbone == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        out_channels = 512
    elif backbone == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet50(weights=weights)
        out_channels = 2048
    else:
        raise ValueError(f"지원하지 않는 backbone: {backbone!r} (resnet18|resnet50)")

    body = nn.Sequential(*list(net.children())[:-2])
    body.out_channels = out_channels

    anchor_generator = AnchorGenerator(
        sizes=(tuple(anchor_sizes),),
        aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes),
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"], output_size=7, sampling_ratio=2
    )

    return FasterRCNN(
        body,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )


# ---------------------------------------------------------------------------
# 3·4단계: RetinaNet
# ---------------------------------------------------------------------------
def build_retinanet(num_classes: int, pretrained: bool = True):
    """COCO 사전학습 RetinaNet(ResNet50-FPN)의 헤드만 교체한 모델.

    분류 헤드뿐 아니라 회귀 헤드도 새로 초기화한다. 원본 스크립트 중 일부는
    분류 헤드만 갈아끼웠는데, 두 헤드의 앵커 수 계산 방식이 어긋날 수 있어
    여기서는 두 헤드 모두 같은 `num_anchors`로 다시 만든다.
    """
    # weights 를 None 으로만 두면 torchvision 이 백본(ResNet50) ImageNet 가중치를
    # 따로 내려받으려 한다. 오프라인 환경에서 완전히 무작위 초기화하려면
    # weights_backbone 도 함께 None 으로 넘겨야 한다.
    weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    weights_backbone = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = retinanet_resnet50_fpn(weights=weights, weights_backbone=weights_backbone)

    in_channels = model.backbone.out_channels
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels, num_anchors, num_classes
    )
    model.head.regression_head = RetinaNetRegressionHead(in_channels, num_anchors)
    return model


def load_checkpoint(
    model: nn.Module, checkpoint_path: str | Path, device: torch.device
) -> nn.Module:
    """체크포인트를 모델에 적재한다.

    원본은 대부분 `strict=False`로 불러왔다. 헤드 교체 후 형태가 안 맞는
    키를 조용히 넘기려는 의도였겠지만, 그 탓에 **가중치가 하나도 안 불러와져도
    에러 없이 통과**한다. 여기서는 strict 로딩을 먼저 시도하고, 실패하면
    무엇이 안 맞는지 출력한 뒤에 비엄격 로딩으로 넘어간다.
    """
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    try:
        model.load_state_dict(state)
        print(f"[체크포인트] strict 로딩 성공: {checkpoint_path}")
    except RuntimeError as exc:
        print(f"[체크포인트] strict 로딩 실패 -> 비엄격 로딩으로 진행: {exc}")
        result = model.load_state_dict(state, strict=False)
        print(
            f"  누락된 키 {len(result.missing_keys)}개 / "
            f"예상 밖 키 {len(result.unexpected_keys)}개"
        )
    return model.to(device)


__all__ = [
    "get_device",
    "build_classifier",
    "build_faster_rcnn",
    "build_retinanet",
    "load_checkpoint",
]
