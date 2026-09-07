"""프로젝트 전역 설정 (클래스 정의, 기본 경로, 상수).

원본 코드에서는 파일마다 클래스 매핑과 경로가 제각각이었다.
(예: `{"aq": 0, "st": 1, "fl": 2}` vs `{"st": 0, "aq": 1, "fl": 2}` vs
 `{"aqua": 0, "string": 1, "floatingr": 2}`)
매핑이 파일마다 다르면 같은 체크포인트를 다른 스크립트로 평가할 때 클래스가
뒤바뀌므로, 여기 한 곳에서만 정의하고 모든 스크립트가 이 값을 임포트해서 쓴다.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# 결함 클래스
# --------------------------------------------------------------------------
# 원본 데이터(CSV/JSON)의 defect_class 값은 아래 세 가지 축약어를 쓴다.
DEFECT_CLASSES: tuple[str, ...] = ("aq", "st", "fl")

DEFECT_CLASS_NAMES: dict[str, str] = {
    "aq": "aqua (해수 얼룩)",
    "st": "string (실 모양 이물질)",
    "fl": "floating (부유물)",
}

# torchvision detection 모델은 라벨 0을 배경(background)으로 예약한다.
# 원본 코드는 실제 결함 클래스에 0을 할당해서 한 클래스가 배경과 충돌했다.
# 여기서는 1부터 시작하도록 바로잡는다.
CLASS_TO_LABEL: dict[str, int] = {name: i + 1 for i, name in enumerate(DEFECT_CLASSES)}
LABEL_TO_CLASS: dict[int, str] = {v: k for k, v in CLASS_TO_LABEL.items()}

# 배경 + 결함 3종
NUM_CLASSES_MULTICLASS: int = len(DEFECT_CLASSES) + 1
# 결함 유형별 단일 클래스 검출기(4단계): 배경 + 결함 1종
NUM_CLASSES_PER_CLASS_DETECTOR: int = 2

# --------------------------------------------------------------------------
# 이미지 / 데이터 상수
# --------------------------------------------------------------------------
IMAGE_SIZE: int = 512  # 원본 김 이미지 해상도 (512x512 RGBA PNG)

# ImageNet 사전학습 백본에 맞춘 정규화 상수
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# 3단계의 "바운딩 박스 정규화" 실험에서 쓴 고정 정사각형 크기.
# 결함 박스가 평균 11px로 매우 작고 크기 편차가 커서, 중심점만 맞추고
# 크기는 고정해버리는 실험을 했다. (실험 상세는 EXPERIMENTS.md 참고)
NORMALIZED_BOX_SIZE: int = 16

# --------------------------------------------------------------------------
# 기본 경로 (모두 CLI 인자로 덮어쓸 수 있다)
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT: Path = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT: Path = PROJECT_ROOT / "outputs"

# 평가 기준값 — 원본 실험에서 계속 쓰던 값을 그대로 유지했다.
DEFAULT_IOU_THRESHOLD: float = 0.75
DEFAULT_SCORE_THRESHOLD: float = 0.5
