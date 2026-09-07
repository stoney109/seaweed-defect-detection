"""박스 좌표까지 함께 변환하는 데이터 증강.

## 원본 코드의 버그 (이 파일이 존재하는 이유)

원본에서는 증강 변환을 이런 식으로 리스트에 담았다.

```python
return [
    T.Compose([T.ToTensor(), T.RandomHorizontalFlip(p=1.0)]),          # 좌우 반전
    T.Compose([T.ToTensor(), T.RandomVerticalFlip(p=1.0)]),            # 상하 반전
    {"transform": T.Compose([...rotate(img, 90)...]), "angle": 90},    # 회전
    ...
]
```

그리고 박스 좌표는 이렇게 변환했다.

```python
def apply_transform_to_boxes(boxes, transform, w, h):
    if isinstance(transform, T.RandomHorizontalFlip):   # (1)
        ...
    elif isinstance(transform, T.RandomVerticalFlip):   # (2)
        ...
    elif isinstance(transform, dict) and "angle" in transform:   # (3)
        ...
    return boxes
```

여기서 리스트에 담긴 것은 `RandomHorizontalFlip` 인스턴스가 아니라 그것을
감싼 `T.Compose` 객체(또는 "angle" 키가 없는 dict)이다. 따라서 (1)과 (2)의
`isinstance` 검사는 **항상 False**가 되고, 반전 증강에서는 박스 좌표가
전혀 변환되지 않은 채 그대로 반환된다.

즉 **이미지는 뒤집혔는데 정답 박스는 원래 자리에 남아 있는 학습 데이터**가
증강본의 상당 비율을 차지했다. 결함 박스가 512px 이미지 안에서 평균 11px에
불과하므로, 뒤집힌 좌표는 사실상 무작위 위치의 오답 라벨이 된다. 회전(3)만
dict 검사에 걸려 정상 동작했다.

이 파일은 변환 종류를 문자열 태그로 명시해 그런 실수가 구조적으로 불가능하게
만든 뒤, 원본의 (정확했던) 회전/반전 좌표 변환 수식을 그대로 옮겨온 것이다.
좌표 수식의 검증은 `tests/test_transforms.py`에 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF

from .config import IMAGENET_MEAN, IMAGENET_STD

TransformKind = Literal["identity", "hflip", "vflip", "rot90", "rot180", "rot270"]

ALL_KINDS: tuple[TransformKind, ...] = (
    "identity",
    "hflip",
    "vflip",
    "rot90",
    "rot180",
    "rot270",
)


def _base_tensor_transform(normalize: bool) -> T.Compose:
    steps: list = [T.ToTensor()]
    if normalize:
        steps.append(T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)))
    return T.Compose(steps)


@dataclass(frozen=True)
class BoxAwareTransform:
    """이미지와 바운딩 박스를 **같은 규칙으로** 변환하는 증강 하나.

    Attributes:
        kind: 변환 종류 태그. 이미지 변환과 박스 변환이 이 태그 하나로 묶여
            있으므로 둘이 어긋날 수 없다.
        normalize: ImageNet 통계로 정규화할지 여부.
    """

    kind: TransformKind = "identity"
    normalize: bool = True

    # -- 이미지 --------------------------------------------------------
    def apply_image(self, img) -> torch.Tensor:
        """PIL 이미지를 텐서로 바꾸고 kind에 맞는 기하 변환을 적용한다."""
        tensor = _base_tensor_transform(self.normalize)(img)

        if self.kind == "identity":
            return tensor
        if self.kind == "hflip":
            return TF.hflip(tensor)
        if self.kind == "vflip":
            return TF.vflip(tensor)
        if self.kind == "rot90":
            return TF.rotate(tensor, 90)
        if self.kind == "rot180":
            return TF.rotate(tensor, 180)
        if self.kind == "rot270":
            return TF.rotate(tensor, 270)
        raise ValueError(f"알 수 없는 변환 종류: {self.kind!r}")

    # -- 박스 ----------------------------------------------------------
    def apply_boxes(self, boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
        """`[N, 4]` (x_min, y_min, x_max, y_max) 박스를 같은 규칙으로 변환한다.

        회전은 torchvision `rotate`와 동일하게 **반시계 방향**이며, 캔버스 크기가
        유지되는(정사각형 입력) 경우를 전제로 한다. 이 프로젝트의 이미지는
        모두 512x512라 이 전제가 성립한다.
        """
        if boxes.numel() == 0:
            return boxes.clone()

        b = boxes.clone().float()

        if self.kind == "identity":
            return b

        if self.kind == "hflip":
            # x -> W - x
            b[:, [0, 2]] = width - boxes[:, [2, 0]]
            return b

        if self.kind == "vflip":
            # y -> H - y
            b[:, [1, 3]] = height - boxes[:, [3, 1]]
            return b

        if self.kind == "rot90":
            # (x, y) -> (y, W - x)  : 반시계 90도
            return torch.stack(
                [
                    boxes[:, 1],           # y_min      -> x_min
                    width - boxes[:, 2],   # W - x_max  -> y_min
                    boxes[:, 3],           # y_max      -> x_max
                    width - boxes[:, 0],   # W - x_min  -> y_max
                ],
                dim=1,
            ).float()

        if self.kind == "rot180":
            b[:, [0, 2]] = width - boxes[:, [2, 0]]
            b[:, [1, 3]] = height - boxes[:, [3, 1]]
            return b

        if self.kind == "rot270":
            # (x, y) -> (H - y, x)  : 반시계 270도 == 시계 90도
            return torch.stack(
                [
                    height - boxes[:, 3],  # H - y_max -> x_min
                    boxes[:, 0],           # x_min     -> y_min
                    height - boxes[:, 1],  # H - y_min -> x_max
                    boxes[:, 2],           # x_max     -> y_max
                ],
                dim=1,
            ).float()

        raise ValueError(f"알 수 없는 변환 종류: {self.kind!r}")

    def __call__(self, img, boxes: torch.Tensor, width: int, height: int):
        return self.apply_image(img), self.apply_boxes(boxes, width, height)


def build_train_transforms(normalize: bool = True) -> list[BoxAwareTransform]:
    """학습용 증강 6종 (원본 + 반전 2종 + 회전 3종)."""
    return [BoxAwareTransform(kind=k, normalize=normalize) for k in ALL_KINDS]


def build_eval_transforms(normalize: bool = True) -> list[BoxAwareTransform]:
    """검증용 — 증강 없이 텐서 변환(+정규화)만."""
    return [BoxAwareTransform(kind="identity", normalize=normalize)]


def build_classification_transform(
    train: bool = False, image_size: int = 224
) -> T.Compose:
    """1단계 이진분류용 전처리 (박스가 없으므로 torchvision 증강을 그대로 쓴다)."""
    steps: list = [T.Resize((image_size, image_size))]
    if train:
        steps += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    steps += [
        T.ToTensor(),
        T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ]
    return T.Compose(steps)
