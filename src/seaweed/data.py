"""데이터셋 클래스 모음.

원본에서는 거의 같은 `SeaweedDataset` 클래스가 파일마다 조금씩 다른 형태로
15번 넘게 복사되어 있었다. 실제로 필요한 형태는 세 가지뿐이라 여기로 모았다.

1. `ClassificationDataset`     — 1단계. 결함/무결함 이진분류.
2. `DetectionDataset`          — 2·3단계. 결함 3종을 한 모델로 검출.
3. `PerClassDetectionDataset`  — 4단계. 결함 1종만 남긴 단일 클래스 검출기용.

## 데이터 형식

CSV (`filtered_seaweed.csv`, `val_seaweed.csv`):

```
image_name,defect_class,top_x,top_y,bot_x,bot_y,label
seaweed_01423.png,fl,206.0,266.0,218.0,278.0,1
```

JSON 라벨은 이미지 한 장당 한 파일이며 같은 필드를 담는다.
`scripts/tools_json_to_csv.py`로 CSV로 변환해서 쓴다.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import CLASS_TO_LABEL, NORMALIZED_BOX_SIZE
from .transforms import BoxAwareTransform, build_eval_transforms

BOX_COLUMNS = ["top_x", "top_y", "bot_x", "bot_y"]


# ---------------------------------------------------------------------------
# 1단계: 이진분류
# ---------------------------------------------------------------------------
class ClassificationDataset(Dataset):
    """결함 이미지(label=1)와 무결함 이미지(label=0)를 함께 로드한다.

    Args:
        defected_dir: 결함 이미지 폴더.
        pure_dir: 무결함 이미지 폴더.
        transform: torchvision 전처리.
        max_pure: 무결함 이미지 사용 개수 상한. 원본 데이터는 결함 3,000장 대
            무결함 255,000장(약 1:85)이라 그대로 쓰면 모델이 전부 "무결함"으로
            찍어버린다. 원본에서는 이 문제를 `pos_weight`만 만지며 풀려고 했고
            결국 나중에 무결함을 1,000장으로 줄인 폴더를 따로 만들었다.
            여기서는 그 subsampling을 옵션으로 노출한다.
    """

    def __init__(
        self,
        defected_dir: str | Path,
        pure_dir: str | Path,
        transform=None,
        max_pure: int | None = None,
        seed: int = 42,
    ) -> None:
        self.transform = transform

        defected = [
            (os.path.join(defected_dir, name), 1)
            for name in sorted(os.listdir(defected_dir))
        ]
        pure = [
            (os.path.join(pure_dir, name), 0)
            for name in sorted(os.listdir(pure_dir))
        ]

        if max_pure is not None and len(pure) > max_pure:
            generator = torch.Generator().manual_seed(seed)
            picked = torch.randperm(len(pure), generator=generator)[:max_pure]
            pure = [pure[i] for i in picked.tolist()]

        self.samples: list[tuple[str, int]] = defected + pure

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def class_balance(self) -> dict[str, int]:
        positives = sum(label for _, label in self.samples)
        return {"defected": positives, "pure": len(self.samples) - positives}


# ---------------------------------------------------------------------------
# 2·3단계: 다중 클래스 검출
# ---------------------------------------------------------------------------
class DetectionDataset(Dataset):
    """결함 3종(aq/st/fl)을 한 모델로 검출하기 위한 데이터셋.

    한 이미지에 여러 박스가 있을 수 있으므로 `image_name` 기준으로 묶어서
    이미지 단위로 인덱싱한다. (원본 일부 파일은 CSV 행 단위로 인덱싱해서
    같은 이미지를 박스 개수만큼 중복 로드했다.)

    Args:
        img_dir: 이미지 폴더.
        csv_file: 라벨 CSV 경로.
        transforms: `BoxAwareTransform` 리스트. 길이가 N이면 데이터셋 크기가
            (이미지 수 x N)이 되고, 각 이미지가 N가지 증강본으로 펼쳐진다.
        normalize_box_size: 지정하면 모든 정답 박스를 원래 박스의 중심을 유지한
            채 이 크기의 정사각형으로 치환한다. 3단계의 "박스 정규화" 실험
            (원본 `retina_boundingbox_normalization.py`)을 옵션으로 옮긴 것.
    """

    def __init__(
        self,
        img_dir: str | Path,
        csv_file: str | Path,
        transforms: list[BoxAwareTransform] | None = None,
        normalize_box_size: int | None = None,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.annotations = pd.read_csv(csv_file)
        self.image_names: list[str] = sorted(self.annotations["image_name"].unique())
        self._groups = {
            name: group for name, group in self.annotations.groupby("image_name")
        }
        self.transforms = transforms or build_eval_transforms()
        self.normalize_box_size = normalize_box_size

    def __len__(self) -> int:
        return len(self.image_names) * len(self.transforms)

    def __getitem__(self, idx: int):
        image_idx, transform_idx = divmod(idx, len(self.transforms))
        img_name = self.image_names[image_idx]
        rows = self._groups[img_name]

        img = Image.open(self.img_dir / img_name).convert("RGB")
        width, height = img.size

        boxes, labels = [], []
        for _, row in rows.iterrows():
            box = [float(row[c]) for c in BOX_COLUMNS]
            if self.normalize_box_size is not None:
                box = _square_box(box, self.normalize_box_size, width, height)
            boxes.append(box)

            defect_class = row["defect_class"]
            if defect_class not in CLASS_TO_LABEL:
                raise ValueError(
                    f"{img_name}: 알 수 없는 defect_class {defect_class!r} "
                    f"(사용 가능: {sorted(CLASS_TO_LABEL)})"
                )
            labels.append(CLASS_TO_LABEL[defect_class])

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        transform = self.transforms[transform_idx]
        image_t = transform.apply_image(img)
        boxes_t = transform.apply_boxes(boxes_t, width, height)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([image_idx]),
            "image_name": img_name,
        }
        return image_t, target


# ---------------------------------------------------------------------------
# 4단계: 결함 유형별 단일 클래스 검출
# ---------------------------------------------------------------------------
class PerClassDetectionDataset(DetectionDataset):
    """결함 한 종류만 남기고, 모든 박스를 라벨 1로 통일한 데이터셋.

    4단계에서 aq/st/fl 각각에 대해 검출기를 따로 학습시킬 때 쓴다.
    원본에서는 `Retina_st_model.py` / `Retina_st_addtional_train.py`가
    소스 코드 안의 문자열(`== "fl"`, `== "st"`)을 직접 고쳐가며 클래스를
    바꿨다. (그래서 파일 이름은 `st`인데 내용은 `fl`을 거르는 상태로 남아 있다.)
    여기서는 생성자 인자로 받는다.
    """

    def __init__(
        self,
        img_dir: str | Path,
        csv_file: str | Path,
        defect_class: str,
        transforms: list[BoxAwareTransform] | None = None,
    ) -> None:
        if defect_class not in CLASS_TO_LABEL:
            raise ValueError(
                f"알 수 없는 defect_class {defect_class!r} "
                f"(사용 가능: {sorted(CLASS_TO_LABEL)})"
            )
        super().__init__(img_dir, csv_file, transforms=transforms)
        self.defect_class = defect_class

        self.annotations = self.annotations[
            self.annotations["defect_class"] == defect_class
        ]
        if self.annotations.empty:
            raise ValueError(f"{csv_file}에 {defect_class!r} 클래스 데이터가 없습니다.")
        self.image_names = sorted(self.annotations["image_name"].unique())
        self._groups = {
            name: group for name, group in self.annotations.groupby("image_name")
        }

    def __getitem__(self, idx: int):
        image, target = super().__getitem__(idx)
        # 단일 클래스 검출기이므로 전부 라벨 1(=해당 결함)로 통일한다.
        target["labels"] = torch.ones_like(target["labels"])
        return image, target


# ---------------------------------------------------------------------------
# 보조 함수
# ---------------------------------------------------------------------------
def _square_box(box: list[float], size: int, width: int, height: int) -> list[float]:
    """박스 중심은 유지한 채 `size x size` 정사각형으로 치환한다."""
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    x_min = max(center_x - size / 2, 0)
    y_min = max(center_y - size / 2, 0)
    x_max = min(x_min + size, width)
    y_max = min(y_min + size, height)
    return [x_min, y_min, x_max, y_max]


def collate_detection(batch):
    """검출 모델용 collate — 이미지와 타깃을 리스트 그대로 넘긴다."""
    images, targets = zip(*batch)
    return list(images), list(targets)


__all__ = [
    "ClassificationDataset",
    "DetectionDataset",
    "PerClassDetectionDataset",
    "collate_detection",
    "NORMALIZED_BOX_SIZE",
]
