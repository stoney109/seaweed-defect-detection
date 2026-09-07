"""JSON 라벨 폴더를 학습용 CSV 한 장으로 변환한다.

원본 라벨은 이미지 한 장당 JSON 한 개다.

```json
{"image_name": "seaweed_01250.png", "defect_class": "st",
 "top_x": 87, "top_y": 104, "bot_x": 100, "bot_y": 116}
```

이걸 모아 `filtered_seaweed.csv` / `val_seaweed.csv` 형태로 만든다.
원본 대응 파일: `SeaWida_1116/json_to_csv.py` (경로가 하드코딩되어 있어 매번
소스를 고쳐 썼다).

실행 예:

    python scripts/tools_json_to_csv.py \\
        --json-dir data/train/train_defected_json \\
        --output   data/filtered_seaweed.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_FIELDS = ["image_name", "defect_class", "top_x", "top_y", "bot_x", "bot_y"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JSON 라벨 -> CSV 변환")
    p.add_argument("--json-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rows: list[dict] = []
    skipped: list[str] = []

    for path in sorted(args.json_dir.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        missing = [k for k in REQUIRED_FIELDS if k not in data]
        if missing:
            skipped.append(f"{path.name} (누락 필드: {missing})")
            continue

        data["label"] = 1  # 결함 존재 여부 플래그 (원본 CSV 형식 유지)
        rows.append(data)

    if not rows:
        raise SystemExit(f"{args.json_dir} 에서 읽을 수 있는 JSON 라벨이 없습니다.")

    df = pd.DataFrame(rows)[REQUIRED_FIELDS + ["label"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"CSV 생성 완료: {args.output} ({len(df)}행)")
    print(f"클래스 분포: {df['defect_class'].value_counts().to_dict()}")
    if skipped:
        print(f"건너뛴 파일 {len(skipped)}개:")
        for item in skipped[:10]:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
