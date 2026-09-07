"""데이터셋 클래스 검증 — 합성 데이터로 라벨 매핑과 증강 확장을 확인한다."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seaweed.config import CLASS_TO_LABEL  # noqa: E402
from seaweed.data import (  # noqa: E402
    DetectionDataset,
    PerClassDetectionDataset,
    collate_detection,
)
from seaweed.transforms import build_eval_transforms, build_train_transforms  # noqa: E402

IMAGE_SIZE = 64


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """이미지 3장 + 라벨 CSV를 만들어 (이미지 폴더, csv 경로)를 돌려준다."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    rows = []
    for i, defect_class in enumerate(["aq", "st", "fl"]):
        name = f"seaweed_{i:05d}.png"
        Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0)).save(img_dir / name)
        rows.append(
            {
                "image_name": name,
                "defect_class": defect_class,
                "top_x": 10,
                "top_y": 12,
                "bot_x": 22,
                "bot_y": 24,
                "label": 1,
            }
        )

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return img_dir, csv_path


def test_background_label_is_reserved(synthetic_dataset) -> None:
    """결함 라벨은 1부터 시작해야 한다 (0은 배경 예약).

    원본 코드는 `{"aq": 0, ...}`처럼 결함에 0을 줘서 배경과 충돌했다.
    """
    assert 0 not in CLASS_TO_LABEL.values()
    assert min(CLASS_TO_LABEL.values()) == 1

    img_dir, csv_path = synthetic_dataset
    ds = DetectionDataset(img_dir, csv_path, transforms=build_eval_transforms())
    for i in range(len(ds)):
        _, target = ds[i]
        assert int(target["labels"].min()) >= 1


def test_dataset_length_scales_with_augmentation(synthetic_dataset) -> None:
    img_dir, csv_path = synthetic_dataset
    plain = DetectionDataset(img_dir, csv_path, transforms=build_eval_transforms())
    augmented = DetectionDataset(img_dir, csv_path, transforms=build_train_transforms())

    assert len(plain) == 3
    assert len(augmented) == 3 * len(build_train_transforms())


def test_every_augmented_index_is_reachable(synthetic_dataset) -> None:
    """증강 인덱싱이 모든 (이미지, 변환) 조합을 정확히 한 번씩 돈다."""
    img_dir, csv_path = synthetic_dataset
    ds = DetectionDataset(img_dir, csv_path, transforms=build_train_transforms())

    seen = []
    for i in range(len(ds)):
        image, target = ds[i]
        assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        seen.append(target["image_name"])

    counts = pd.Series(seen).value_counts()
    assert set(counts.values) == {len(build_train_transforms())}


def test_per_class_dataset_filters_and_relabels(synthetic_dataset) -> None:
    img_dir, csv_path = synthetic_dataset
    ds = PerClassDetectionDataset(
        img_dir, csv_path, defect_class="st", transforms=build_eval_transforms()
    )

    assert len(ds) == 1  # st 클래스 이미지 1장만
    _, target = ds[0]
    assert target["labels"].tolist() == [1]  # 단일 클래스 검출기는 전부 라벨 1


def test_per_class_dataset_rejects_unknown_class(synthetic_dataset) -> None:
    img_dir, csv_path = synthetic_dataset
    with pytest.raises(ValueError):
        PerClassDetectionDataset(img_dir, csv_path, defect_class="xx")


def test_unknown_defect_class_in_csv_raises(tmp_path: Path) -> None:
    """CSV에 예상 밖 클래스가 있으면 명확한 에러를 낸다.

    원본 콜랩 노트북(`Untitled6.ipynb`)은 라벨 매핑 키를 `aqua/string/floatingr`로
    적어놨는데 실제 CSV 값은 `aq/st/fl`이라 여기서 터졌다. 게다가 에러 메시지가
    참조하는 변수명까지 틀려서(`defect_class` 미정의) 원래 에러 대신
    NameError가 났다.
    """
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)).save(img_dir / "a.png")

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {
                "image_name": "a.png",
                "defect_class": "aqua",  # 잘못된 값
                "top_x": 1,
                "top_y": 1,
                "bot_x": 5,
                "bot_y": 5,
                "label": 1,
            }
        ]
    ).to_csv(csv_path, index=False)

    ds = DetectionDataset(img_dir, csv_path, transforms=build_eval_transforms())
    with pytest.raises(ValueError, match="알 수 없는 defect_class"):
        _ = ds[0]


def test_normalize_box_size_produces_square_boxes(synthetic_dataset) -> None:
    img_dir, csv_path = synthetic_dataset
    ds = DetectionDataset(
        img_dir, csv_path, transforms=build_eval_transforms(), normalize_box_size=16
    )
    _, target = ds[0]
    box = target["boxes"][0]
    assert float(box[2] - box[0]) == pytest.approx(16.0)
    assert float(box[3] - box[1]) == pytest.approx(16.0)


def test_collate_keeps_variable_length_targets(synthetic_dataset) -> None:
    img_dir, csv_path = synthetic_dataset
    ds = DetectionDataset(img_dir, csv_path, transforms=build_eval_transforms())
    images, targets = collate_detection([ds[0], ds[1]])

    assert isinstance(images, list) and len(images) == 2
    assert isinstance(targets, list) and len(targets) == 2
    assert all(isinstance(t["boxes"], torch.Tensor) for t in targets)
