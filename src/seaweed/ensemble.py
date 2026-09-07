"""4단계 앙상블 — 결함 유형별 단일 클래스 검출기 3개를 묶어 하나로 예측한다.

## 구조

aq / st / fl 각각에 대해 "배경 vs 해당 결함" 2클래스 RetinaNet을 따로 학습한 뒤,
추론 시 세 모델의 예측을 모두 모아 **신뢰도가 가장 높은 박스 하나**를 고른다.
이때 그 박스를 낸 모델이 곧 예측된 결함 유형이 된다.

한 이미지에 결함이 하나뿐인 이 데이터셋 특성(3,000장 / 박스 3,000개) 때문에
"가장 확신하는 하나만 남긴다"는 단순한 규칙이 성립한다.

## 원본 대비 고친 것

1. **평가 시 절반이 누락되던 문제**
   원본 `dddddddddd.py`는 `DataLoader(batch_size=2)`로 만들어놓고 루프 안에서
   `images[0]`만 사용했다. 배치의 두 번째 이미지는 매번 버려져서 검증셋 750장
   중 절반만 평가된 셈이다. 여기서는 배치 전체를 순회한다.

2. **시각화용 박스 확대가 지표를 오염시키던 문제**
   원본 `get_predictions()`는 너무 작아 눈에 안 보이는 박스를 최소 10px로
   강제로 키웠는데, 그 확대된 박스가 그대로 IoU 계산에 들어갔다. 정답 박스가
   평균 11px이라 이 보정은 IoU를 크게 왜곡한다. 확대는 시각화 함수에서만
   하도록 분리했다.

3. **CSV 재읽기**
   원본은 이미지 한 장을 꺼낼 때마다 `pd.read_csv()`로 전체 CSV를 다시 읽었다.
   데이터셋 생성 시 한 번만 읽도록 바꿨다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import NUM_CLASSES_PER_CLASS_DETECTOR
from .models import build_retinanet, load_checkpoint


class DefectEnsemble:
    """결함 유형별 검출기 묶음.

    Args:
        checkpoints: `{"aq": 경로, "st": 경로, "fl": 경로}` 형태의 체크포인트 맵.
        device: 추론 디바이스.
    """

    def __init__(self, checkpoints: dict[str, str | Path], device: torch.device) -> None:
        if not checkpoints:
            raise ValueError("체크포인트가 하나도 지정되지 않았습니다.")

        self.device = device
        self.defect_classes: list[str] = list(checkpoints.keys())
        self.models: list[torch.nn.Module] = []

        for defect_class, path in checkpoints.items():
            model = build_retinanet(
                num_classes=NUM_CLASSES_PER_CLASS_DETECTOR, pretrained=False
            )
            model = load_checkpoint(model, path, device)
            model.eval()
            self.models.append(model)
            print(f"[앙상블] {defect_class} 검출기 로드 완료: {path}")

    @torch.no_grad()
    def predict(self, image: torch.Tensor, score_threshold: float = 0.5):
        """이미지 한 장에 대해 세 모델을 돌리고 가장 확신하는 박스 하나를 고른다.

        Returns:
            (box, score, defect_class) — 임계값을 넘는 예측이 없으면
            `(None, 0.0, None)`.
        """
        image = image.to(self.device)

        best_box, best_score, best_class = None, 0.0, None

        for defect_class, model in zip(self.defect_classes, self.models):
            output = model([image])[0]
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()

            keep = scores >= score_threshold
            boxes, scores = boxes[keep], scores[keep]
            if len(scores) == 0:
                continue

            top = int(np.argmax(scores))
            if float(scores[top]) > best_score:
                best_box = boxes[top]
                best_score = float(scores[top])
                best_class = defect_class

        return best_box, best_score, best_class

    def save(self, path: str | Path) -> None:
        """세 모델의 state_dict를 클래스 이름과 함께 한 파일로 저장한다.

        원본 `save_ensemble_models()`는 state_dict 리스트만 저장해서 어떤
        인덱스가 어떤 결함 유형인지 파일 안에 남지 않았다(코드의 경로 순서에만
        의존). 여기서는 클래스 이름을 함께 저장한다.
        """
        payload = {
            "defect_classes": self.defect_classes,
            "state_dicts": [m.state_dict() for m in self.models],
        }
        torch.save(payload, path)
        print(f"[앙상블] 저장 완료: {path}")


__all__ = ["DefectEnsemble"]
