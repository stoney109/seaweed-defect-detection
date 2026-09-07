"""증강 시 이미지와 박스가 실제로 같이 움직이는지 검증한다.

이 테스트가 이 저장소에서 가장 중요한 테스트다. 원본 코드에서 반전 증강은
이미지만 뒤집고 박스는 그대로 두는 버그가 있었고(자세한 설명은
`src/seaweed/transforms.py` 상단), 그 버그는 눈으로는 보이지 않는다.
학습 데이터가 조용히 오염될 뿐이다.

검증 방법: 박스 영역에만 밝은 값을 칠한 합성 이미지를 만들고, 변환 후에도
변환된 박스 좌표가 밝은 영역을 정확히 가리키는지 픽셀로 확인한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seaweed.transforms import ALL_KINDS, BoxAwareTransform  # noqa: E402

IMAGE_SIZE = 64
BOX = [8.0, 20.0, 16.0, 30.0]  # x_min, y_min, x_max, y_max (비대칭 위치·크기)


def make_image_with_marked_box(box: list[float], size: int = IMAGE_SIZE) -> Image.Image:
    """박스 내부만 흰색, 나머지는 검은색인 합성 이미지."""
    tensor = torch.zeros(3, size, size)
    x1, y1, x2, y2 = (int(v) for v in box)
    tensor[:, y1:y2, x1:x2] = 1.0
    array = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(array)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_box_follows_image(kind: str) -> None:
    """변환 후 박스 좌표가 여전히 흰 영역을 정확히 감싸는가."""
    image = make_image_with_marked_box(BOX)
    boxes = torch.tensor([BOX], dtype=torch.float32)

    transform = BoxAwareTransform(kind=kind, normalize=False)
    out_image = transform.apply_image(image)
    out_boxes = transform.apply_boxes(boxes, IMAGE_SIZE, IMAGE_SIZE)

    x1, y1, x2, y2 = (int(round(v)) for v in out_boxes[0].tolist())
    assert x1 < x2 and y1 < y2, f"{kind}: 변환 후 박스가 뒤집혔다 {out_boxes[0].tolist()}"

    channel = out_image[0]
    inside = channel[y1:y2, x1:x2]

    # 1) 변환된 박스 안은 전부 흰색이어야 한다.
    assert inside.numel() > 0, f"{kind}: 박스 영역이 비었다"
    assert inside.min() > 0.5, (
        f"{kind}: 변환된 박스 안에 검은 픽셀이 있다 "
        f"(= 박스가 이미지와 함께 움직이지 않았다)"
    )

    # 2) 이미지 전체의 흰 픽셀 수와 박스 넓이가 일치해야 한다
    #    (박스가 흰 영역의 일부만 덮고 있지 않은지 확인).
    white_pixels = int((channel > 0.5).sum())
    assert white_pixels == inside.numel(), (
        f"{kind}: 흰 픽셀 {white_pixels}개 vs 박스 넓이 {inside.numel()} — "
        f"박스가 흰 영역을 정확히 덮지 못했다"
    )


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_box_stays_in_bounds(kind: str) -> None:
    """변환된 박스가 이미지 밖으로 나가지 않는가."""
    boxes = torch.tensor([BOX], dtype=torch.float32)
    out = BoxAwareTransform(kind=kind).apply_boxes(boxes, IMAGE_SIZE, IMAGE_SIZE)
    assert out.min() >= 0
    assert out.max() <= IMAGE_SIZE


def test_identity_is_noop() -> None:
    boxes = torch.tensor([BOX], dtype=torch.float32)
    out = BoxAwareTransform(kind="identity").apply_boxes(boxes, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.allclose(out, boxes)


def test_double_flip_returns_to_original() -> None:
    """같은 반전을 두 번 적용하면 원래 좌표로 돌아와야 한다."""
    boxes = torch.tensor([BOX], dtype=torch.float32)
    for kind in ("hflip", "vflip", "rot180"):
        transform = BoxAwareTransform(kind=kind)
        once = transform.apply_boxes(boxes, IMAGE_SIZE, IMAGE_SIZE)
        twice = transform.apply_boxes(once, IMAGE_SIZE, IMAGE_SIZE)
        assert torch.allclose(twice, boxes), f"{kind}를 두 번 적용했는데 원위치가 아니다"


def test_rotation_four_times_returns_to_original() -> None:
    """90도 회전을 네 번 하면 제자리로 돌아와야 한다."""
    boxes = torch.tensor([BOX], dtype=torch.float32)
    transform = BoxAwareTransform(kind="rot90")
    current = boxes
    for _ in range(4):
        current = transform.apply_boxes(current, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.allclose(current, boxes)


def test_empty_boxes_are_handled() -> None:
    empty = torch.zeros((0, 4), dtype=torch.float32)
    for kind in ALL_KINDS:
        out = BoxAwareTransform(kind=kind).apply_boxes(empty, IMAGE_SIZE, IMAGE_SIZE)
        assert out.shape == (0, 4)


def test_unknown_kind_raises() -> None:
    boxes = torch.tensor([BOX], dtype=torch.float32)
    with pytest.raises(ValueError):
        BoxAwareTransform(kind="rot45").apply_boxes(boxes, IMAGE_SIZE, IMAGE_SIZE)  # type: ignore[arg-type]
