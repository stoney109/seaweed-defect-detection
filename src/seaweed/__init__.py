"""김(seaweed) 이물질 검출 프로젝트 — 공용 모듈.

WIDA 소학회 2024-2학기 프로젝트. 512x512 김 이미지에서 이물질(결함)을
검출하는 모델을 이진분류 → 객체탐지 → 결함 유형별 앙상블 순으로 발전시킨
실험 기록을 정리한 코드다. 실험 연대기는 `EXPERIMENTS.md` 참고.
"""

from .config import (
    CLASS_TO_LABEL,
    DEFECT_CLASSES,
    LABEL_TO_CLASS,
    NUM_CLASSES_MULTICLASS,
    NUM_CLASSES_PER_CLASS_DETECTOR,
)

__all__ = [
    "DEFECT_CLASSES",
    "CLASS_TO_LABEL",
    "LABEL_TO_CLASS",
    "NUM_CLASSES_MULTICLASS",
    "NUM_CLASSES_PER_CLASS_DETECTOR",
]

__version__ = "1.0.0"
