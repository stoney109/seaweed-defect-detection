#!/usr/bin/env bash
# 전체 파이프라인 스모크 테스트.
#
# 합성 데이터(128x128 작은 이미지 수십 장)를 만들어 8개 스크립트를 전부 1~2에폭씩
# 돌려본다. 성능을 보려는 게 아니라 **모든 경로가 실제로 실행되는지** 확인하는 것이
# 목적이다. 무작위 초기화(`--no-pretrained`)를 쓰므로 지표는 0으로 나오는 게 정상이다.
#
# CPU로 약 8~10분 걸린다.
#
#   bash tests/smoke_test.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-/tmp/seaweed_smoke}"

cd "$REPO_ROOT"
rm -rf "$WORK"
mkdir -p "$WORK"

echo "작업 폴더: $WORK"
echo

# ---------------------------------------------------------------------------
echo "[1/8] 합성 데이터 생성"
python3 - "$WORK" <<'PY'
import json, random, sys
from pathlib import Path
from PIL import Image, ImageDraw

random.seed(0)
root = Path(sys.argv[1])
for split, n_def, n_pure in [("train", 12, 8), ("validation", 6, 4)]:
    d_img = root / split / f"{split}_defected_dataset"; d_img.mkdir(parents=True, exist_ok=True)
    d_json = root / split / f"{split}_defected_json"; d_json.mkdir(parents=True, exist_ok=True)
    p_img = root / split / f"{split}_pure_dataset"; p_img.mkdir(parents=True, exist_ok=True)

    for i in range(n_def):
        name = f"seaweed_{split}_{i:03d}.png"
        img = Image.new("RGB", (128, 128), (30, 60, 30))
        x, y = random.randint(10, 100), random.randint(10, 100)
        size = random.randint(6, 14)
        ImageDraw.Draw(img).rectangle([x, y, x + size, y + size], fill=(220, 220, 200))
        img.save(d_img / name)
        (d_json / f"seaweed_{split}_{i:03d}.json").write_text(
            json.dumps({
                "image_name": name,
                "defect_class": ["aq", "st", "fl"][i % 3],
                "top_x": x, "top_y": y, "bot_x": x + size, "bot_y": y + size,
            }), encoding="utf-8")

    for i in range(n_pure):
        Image.new("RGB", (128, 128), (30, 60, 30)).save(p_img / f"pure_{split}_{i:03d}.png")
print("  합성 데이터 생성 완료")
PY

# ---------------------------------------------------------------------------
echo "[2/8] JSON -> CSV 변환"
python3 scripts/tools_json_to_csv.py --json-dir "$WORK/train/train_defected_json" --output "$WORK/train.csv"
python3 scripts/tools_json_to_csv.py --json-dir "$WORK/validation/validation_defected_json" --output "$WORK/val.csv"

echo "[3/8] 데이터셋 통계"
python3 scripts/tools_analyze_dataset.py --csv "$WORK/train.csv" | head -6

echo "[4/8] 1단계 이진분류"
python3 scripts/stage1_train_classifier.py \
  --train-defected "$WORK/train/train_defected_dataset" \
  --train-pure     "$WORK/train/train_pure_dataset" \
  --val-defected   "$WORK/validation/validation_defected_dataset" \
  --val-pure       "$WORK/validation/validation_pure_dataset" \
  --backbone resnet18 --no-pretrained --epochs 2 --batch-size 4 --num-workers 0 \
  --output "$WORK/out/stage1.pth"

echo "[5/8] 2단계 Faster R-CNN (증강 on)"
python3 scripts/stage2_train_faster_rcnn.py \
  --train-images "$WORK/train/train_defected_dataset" --train-csv "$WORK/train.csv" \
  --val-images   "$WORK/validation/validation_defected_dataset" --val-csv "$WORK/val.csv" \
  --backbone resnet18 --no-pretrained --epochs 1 --batch-size 2 --num-workers 0 --augment \
  --output "$WORK/out/stage2.pth"

echo "[6/8] 3단계 RetinaNet (박스 정규화 실험 경로)"
python3 scripts/stage3_train_retinanet.py \
  --train-images "$WORK/train/train_defected_dataset" --train-csv "$WORK/train.csv" \
  --val-images   "$WORK/validation/validation_defected_dataset" --val-csv "$WORK/val.csv" \
  --no-pretrained --epochs 1 --batch-size 2 --num-workers 0 --normalize-box-size 16 \
  --output-dir "$WORK/out/stage3"

echo "[7/8] 4단계 유형별 학습 + 이어학습"
python3 scripts/stage4_train_per_class.py --defect-class aq \
  --train-images "$WORK/train/train_defected_dataset" --train-csv "$WORK/train.csv" \
  --val-images   "$WORK/validation/validation_defected_dataset" --val-csv "$WORK/val.csv" \
  --no-pretrained --epochs 1 --batch-size 2 --num-workers 0 --augment \
  --output-dir "$WORK/out/stage4/aq"

python3 scripts/stage4_train_per_class.py --defect-class aq \
  --train-images "$WORK/train/train_defected_dataset" --train-csv "$WORK/train.csv" \
  --val-images   "$WORK/validation/validation_defected_dataset" --val-csv "$WORK/val.csv" \
  --no-pretrained --epochs 1 --batch-size 2 --num-workers 0 \
  --resume "$WORK/out/stage4/aq/retinanet_aq_epoch_01.pth" --start-epoch 2 \
  --output-dir "$WORK/out/stage4/aq"

echo "[8/8] 앙상블 평가 + 시각화"
CKPT="$WORK/out/stage4/aq/retinanet_aq_epoch_02.pth"
python3 scripts/stage4_eval_ensemble.py \
  --val-images "$WORK/validation/validation_defected_dataset" --val-csv "$WORK/val.csv" \
  --checkpoint "aq=$CKPT" --checkpoint "st=$CKPT" --checkpoint "fl=$CKPT" \
  --num-workers 0 --save-ensemble "$WORK/out/ensemble.pth" --report "$WORK/out/report.json"

python3 scripts/tools_visualize_predictions.py \
  --val-images "$WORK/validation/validation_defected_dataset" --val-csv "$WORK/val.csv" \
  --checkpoint "aq=$CKPT" --checkpoint "st=$CKPT" --checkpoint "fl=$CKPT" \
  --num-images 3 --output-dir "$WORK/out/vis"

echo
echo "스모크 테스트 통과 — 8개 스크립트 전부 정상 실행."
echo "산출물: $WORK/out"
