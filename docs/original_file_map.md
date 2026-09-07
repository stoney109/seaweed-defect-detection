# 원본 파일 대응표

정리 전 원본은 콜랩 노트북 3개 + 로컬/PC방 프로젝트 폴더 5개에 파이썬 파일
45개(약 7,800줄)로 흩어져 있었다. 각각이 정리본의 어디로 갔는지 정리한 표다.

폴더별 성격:

| 원본 폴더 | 시기 | 환경 | 성격 |
|-----------|------|------|------|
| `SeaWeed_Pj/` | 2024-09~10 | 로컬 PyCharm | 1단계 분류 + 2단계 검출 초기 |
| `SeaWida/` | 2024-11 | 로컬 PyCharm | 3단계 RetinaNet + 분석 도구 |
| `SeaWida_1116/` | 2024-11-16~ | 로컬 PyCharm | 3단계 RetinaNet 개선 |
| `SeaWida_string/` | 2024-11-18~ | PC방 | 4단계 유형별 검출기 |
| `SeaWida_string_1123/` | 2024-11-23~ | PC방 | 4단계 앙상블 (최종) |

`SeaWida_string`과 `SeaWida_string_1123`은 폴더를 통째로 복사해 이어간 것이라
`Retina_st_model.py`, `Retina_st_addtional_train.py`, `visual_False_iou_bound.py`,
`visualization_width_height.py`, `retina_boundingbox_normalization.py`가 두 폴더에
바이트 단위로 동일하게 존재한다.

---

## 1단계 — 이진분류

| 원본 | 정리본 |
|------|--------|
| `Seaweed_ResNet.ipynb` (콜랩) | `scripts/stage1_train_classifier.py` |
| `SeaWeed_Pj/Previous_Files/ResNet.py` | 〃 (`--backbone resnet18`, 조기 종료 개념) |
| `SeaWeed_Pj/Previous_Files/add_preprocess.py` | 〃 (`--augment`) |
| `SeaWeed_Pj/Previous_Files/fliped_image.py` | 〃 (`--augment`) |
| `SeaWeed_Pj/Previous_Files/parameter_change.py` | 미포함 — 그리드서치 스크립트. 결과가 `parameter_fixed.py`에 반영됨 |
| `SeaWeed_Pj/Previous_Files/parameter_fixed.py` | 〃 (기본 하이퍼파라미터의 출처) |
| `SeaWeed_Pj/1003_01_new_model.py` | 〃 (`--backbone resnet50 --pos-weight 5.0 --threshold 0.3`) |
| `SeaWeed_Pj/dkd.py` | 〃 (`--pos-weight 0.14` 실험) |
| `SeaWeed_Pj/tjgusdlrk_model.py` | 〃 (무결함 증강 → `--max-pure`로 대체) |
| `SeaWeed_Pj/validation.py` | 〃 (학습 루프에 검증 내장) |
| `SeaWeed_Pj/Previous_Files/validate.py` | 〃 |
| `SeaWeed_Pj/d.py` | **미포함** — 2줄짜리 낙서. `Sm.join('dddd')`로 정의되지 않은 이름을 참조해 실행 자체가 불가능 |

## 2단계 — Faster R-CNN

| 원본 | 정리본 |
|------|--------|
| `Untitled1.ipynb` (콜랩) | `scripts/stage2_train_faster_rcnn.py` |
| `SeaWeed_Pj/dlddl.py` | 〃 (JSON 라벨 → `tools_json_to_csv.py`로 CSV 변환 후 사용) |
| `SeaWeed_Pj/seawida.py` | 〃 (앵커 축소, IoU 평가 — 이 버전이 기준) |
| `seawida.py` (최상위) | 〃 (위와 거의 동일한 사본) |

## 3단계 — RetinaNet

| 원본 | 정리본 |
|------|--------|
| `Untitled6.ipynb` (콜랩) | `scripts/stage3_train_retinanet.py` — 실행은 실패했음 (EXPERIMENTS.md 참고) |
| `SeaWida_1116/RetinaNet.py` | 〃 |
| `SeaWida_1116/RetinaNet_csv_ver.py` | 〃 |
| `SeaWida_1116/RetinaNet_saveRepeat.py` | 〃 (에폭별 체크포인트 저장) |
| `SeaWida_1116/RetinaNet_validate.py` | 〃 |
| `SeaWida_1116/Retina_data_augmentation.py` | 〃 (`--augment`) |
| `SeaWida_1116/Retina_new_additional.py` | 〃 (`--resume`) |
| `SeaWida_1116/Retina_addtional.py` | 〃 (`--resume`, 회전 증강 포함) |
| `SeaWida_1116/precision_recall_validate.py` | `src/seaweed/metrics.py` |
| `SeaWida/retina_boundingbox_normalization.py` | 〃 `--normalize-box-size` |
| `SeaWida/validation_new.py` | `src/seaweed/metrics.py` |
| `SeaWida_1116/json_to_csv.py` | `scripts/tools_json_to_csv.py` |
| `SeaWida_1116/flip_rotate_test.py` | `tests/test_transforms.py` (눈으로 확인 → 자동 검증으로 대체) |

## 분석·시각화 도구

| 원본 | 정리본 |
|------|--------|
| `SeaWida/visualization_width_height.py` | `scripts/tools_analyze_dataset.py` |
| `SeaWida/ddd..py` | 〃 (박스 면적 히스토그램만 있는 축약판) |
| `SeaWida_string_1123/visual_less75_bbox_size.py` | 〃 |
| `SeaWida/visualization_iou.py` | `scripts/stage4_eval_ensemble.py` (IoU 집계) |
| `SeaWida/visualization_iou_normalization.py` | 〃 |
| `SeaWida_string/visualization_iou_normalization.py` | 〃 (중심거리 분석이 추가된 버전) |
| `SeaWida/visual_False_iou_bound.py` | `scripts/tools_visualize_predictions.py --only-failures` |
| `SeaWida/validation_zero_bbox.py` | 〃 |
| `SeaWida/validate_class_ratio.py` | `scripts/stage4_eval_ensemble.py` (클래스별 집계) |
| `SeaWida_string_1123/counting_iou_low_75.py` | 〃 |
| `SeaWida_string_1123/visual_bessssst_iou_hist.py` | 〃 |

## 4단계 — 유형별 검출기 + 앙상블

| 원본 | 정리본 |
|------|--------|
| `SeaWida_string/Retina_st_model.py` | `scripts/stage4_train_per_class.py` (`--defect-class`) |
| `SeaWida_string/Retina_st_addtional_train.py` | 〃 (`--resume --start-epoch`) |
| `SeaWida_string_1123/Retina_st_model.py` | 〃 (위와 동일 파일) |
| `SeaWida_string_1123/Retina_st_addtional_train.py` | 〃 (위와 동일 파일) |
| `SeaWida_string_1123/model_ensemble.py` | `scripts/stage4_eval_ensemble.py` |
| `SeaWida_string_1123/ssshyun_is_the_best.py` | 〃 (그 이전 버전) |
| `SeaWida_string_1123/dddddddddd.py` | 〃 (**가장 발전된 최종본** — 이 버전이 기준) |
| `SeaWida_string_1123/st_models_pick/st_model_validation.txt` | `results/stage4_string_detector_metrics.md` |
| `SeaWida_string_1123/st_models_pick/README.txt` | 〃 (체크포인트 선정 근거) |

## 미포함 파일

| 원본 | 이유 |
|------|------|
| `SeaWeed_Pj/d.py` | 2줄짜리 낙서, 실행 불가 (`Sm` 미정의) |
| `SeaWeed_Pj/Previous_Files/parameter_change.py` | 162개 조합 그리드서치. 결론이 다른 파일에 반영되어 있고, 근본 원인이 하이퍼파라미터가 아니었음 |
| `*/__init__.py` (빈 파일) | 패키지 구조 재편으로 불필요 |
| `*/.idea/`, `__pycache__/` | IDE 설정 / 캐시 |
| `*.pth` (체크포인트 60여 개) | 용량 문제로 제외. `.gitignore`에 포함 |
| `dataset/`, `datasets/` | 이미지 30만 장 이상. 저장소에 넣지 않음 |
