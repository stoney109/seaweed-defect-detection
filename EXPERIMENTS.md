# 실험 연대기

2024 DATA·AI 분석 경진대회(KISTI) 참가 기간인 2024년 9월부터 11월까지 진행한
실험 기록. 원본 코드의 실행 시각 메타데이터와 체크포인트 파일명, 폴더 구조에서
복원한 것이다.

대회 과제는 ① 품질 등급 분류(결함/무결함) ② 결함 검출 ③ 결함의 위치와 유형 식별,
이렇게 세 가지를 요구했다. 아래 4단계는 그 요구사항을 순서대로 밟아간 과정이다.
1단계가 ①, 2단계부터가 ②·③에 해당하고, 특히 "유형까지" 식별하라는 조건이
4단계의 유형별 검출기 + 앙상블 구조를 낳았다.

작업 환경이 세 군데로 나뉘어 있었다.

- **Google Colab** — 초기 시도. 무료 GPU 세션이 끊기고 데이터 업로드가 느려서 오래 못 썼다.
- **로컬 PyCharm** — 3~4단계 대부분. `SeaWeed_Pj`, `SeaWida`, `SeaWida_1116`.
- **PC방** — 로컬 GPU로 감당이 안 될 때. `SeaWida_string`, `SeaWida_string_1123`.

---

## 1단계 — 이진분류 (2024-09-26 ~ 10월 초)

**"이 김에 이물질이 있는가?"만 판단.** 위치는 찾지 않는다. 대회 요구사항 중
"김의 품질에 따라 등급을 분류(결함/무결함)"에 해당하는 부분이다.

사전학습 ResNet의 마지막 층을 출력 1개로 바꾸고 `BCEWithLogitsLoss`로 학습했다.
콜랩 노트북 `Seaweed_ResNet.ipynb`(2024-09-26)가 첫 시도고, 이후 로컬
`SeaWeed_Pj/`에서 같은 코드를 계속 변형했다.

`SeaWeed_Pj/`에 남아 있던 변형들:

| 원본 파일 | 무엇을 바꿔봤나 |
|-----------|-----------------|
| `Previous_Files/ResNet.py` | ResNet18 + 조기 종료(patience=2), `pos_weight=3.0` |
| `Previous_Files/add_preprocess.py` | ResNet50, 랜덤 반전·30도 회전 증강 추가 |
| `Previous_Files/fliped_image.py` | 좌우반전 데이터로 기존 모델 이어학습 |
| `Previous_Files/parameter_change.py` | 하이퍼파라미터 그리드서치 (162개 조합) |
| `Previous_Files/parameter_fixed.py` | 그리드서치 결과를 고정값으로 |
| `1003_01_new_model.py` | ResNet50, `pos_weight=5.0`, threshold 0.3 |
| `dkd.py` | `pos_weight=0.14`로 극단 실험 |
| `tjgusdlrk_model.py` | 무결함 이미지만 반전 증강해 개수 늘리기 |
| `validation.py`, `Previous_Files/validate.py` | 저장된 모델 재평가 전용 |

### 여기서 막힌 것

`parameter_change.py` 맨 위에 이런 주석이 남아 있다.

```python
# 이건 혹시 파라미터 문제일까봐 해봤어요
# 그러나 recall이 계속 높게 나오더라고요...
```

원인은 하이퍼파라미터가 아니라 **데이터 불균형**이었다. 학습 폴더에 결함
이미지 3,000장, 무결함 이미지 255,000장(1:85)이 들어 있었다. `pos_weight`를
0.14부터 5.0까지 바꿔가며 실험한 것은 이 불균형을 손실 함수 쪽에서 상쇄해
보려는 시도였는데, 85배 차이를 가중치만으로 메우기는 어렵다.

나중에 `SeaWida_1116/dataset/train/train_pure_dataset`에는 무결함이 1,000장만
들어 있다. 어느 시점엔가 무결함을 직접 솎아냈다는 뜻이다. 정리본에서는 그
subsampling을 `--max-pure` 인자로 노출했다.

> **정리본 대응:** `scripts/stage1_train_classifier.py`
> (위 9개 변형을 CLI 인자 `--backbone/--pos-weight/--threshold/--max-pure/--augment`로 통합)

---

## 2단계 — Faster R-CNN (2024-10-11)

이진분류로는 "이 김 어딘가에 이물질이 있다"까지만 알 수 있다. 대회 요구사항이
"결함의 위치와 유형을 정확히 식별"까지였으므로 객체탐지로 넘어갔다.

- `SeaWeed_Pj/dlddl.py` — JSON 라벨을 직접 읽는 첫 버전. 유효하지 않은 박스가
  나오면 `[0,0,1,1]` 더미 박스를 넣는 임시 처리가 있다.
- `SeaWeed_Pj/seawida.py` — CSV 라벨 사용, `WeightedRandomSampler`로 클래스
  불균형 보정, IoU 평가 함수 추가. 백본을 ResNet18로 낮추고 앵커를
  `(32,64,128,256,512)` → `(16,32,64,128,256)`으로 내렸다.
- 콜랩 `Untitled1.ipynb` — 같은 구조를 콜랩에서 돌려본 것. 마지막 학습 셀은
  실행되지 않은 채로 저장되어 있다.

### 여기서 배운 것

앵커 크기를 내린 것이 핵심이다. 결함 박스가 평균 11px인데 기본 앵커의 최소
크기가 32px이라, 그대로는 정답과 IoU가 겹치는 앵커가 아예 없어서 학습 신호가
생기지 않는다.

> **정리본 대응:** `scripts/stage2_train_faster_rcnn.py`, `src/seaweed/models.py::build_faster_rcnn`

---

## 3단계 — RetinaNet + 증강 (2024-11-12 ~ 11-18)

Faster R-CNN도 작은 결함에서 Recall이 오르지 않아, 작은 객체에 강하다고 알려진
1-stage 검출기 RetinaNet(ResNet50-FPN, COCO 사전학습)으로 갈아탔다.

`SeaWida_1116/`에 개선 과정이 파일 단위로 남아 있다.

| 순서 | 원본 파일 | 추가된 것 |
|:----:|-----------|-----------|
| 1 | `RetinaNet.py` | 기본형. JSON 라벨, 5에폭 |
| 2 | `RetinaNet_csv_ver.py` | CSV 라벨로 전환 |
| 3 | `RetinaNet_saveRepeat.py` | 에폭마다 체크포인트 저장, 클래스 검증 평가 추가 |
| 4 | `RetinaNet_validate.py` | 검증 루프 분리 |
| 5 | `Retina_data_augmentation.py` | 반전·회전 증강 도입 |
| 6 | `Retina_new_additional.py` | 저장된 체크포인트에서 이어학습 (반전만) |
| 7 | `Retina_addtional.py` | 회전 90/180/270도까지 확장, 이어학습 |
| — | `precision_recall_validate.py` | Precision/Recall/mAP 검증 전용 |
| — | `flip_rotate_test.py` | 증강 결과를 눈으로 확인하는 시각화 |

`SeaWida/`에는 이 시기의 분석 도구들이 쌓여 있다.

- `visualization_iou.py` — 예측 IoU 분포 히스토그램
- `visual_False_iou_bound.py` — IoU가 0인 실패 사례를 이미지로 확인
- `visualization_width_height.py`, `ddd..py` — 결함 박스 크기 분포
- `validate_class_ratio.py` — 클래스별 오탐 분석
- `validation_zero_bbox.py` — 아예 못 잡은 케이스의 박스 크기 분포
- `retina_boundingbox_normalization.py` — **박스 정규화 실험**

### 박스 정규화 실험

`retina_boundingbox_normalization.py`가 이 단계에서 가장 독창적인 시도다.
정답 박스의 중심만 유지한 채 크기를 전부 16×16 정사각형으로 바꿔서 학습시켰다.

의도는 이렇게 읽힌다 — 결함 박스는 3px부터 21px까지 크기가 제각각인데, 어차피
공정에서 필요한 건 "여기에 이물질이 있다"는 위치 정보다. 크기까지 정확히 맞추라고
요구하면 회귀 헤드가 어려워지니, 크기를 상수로 고정하고 위치만 학습시키자는 것이다.

> **정리본 대응:** `scripts/stage3_train_retinanet.py`
> (7개 변형을 `--augment/--resume/--start-epoch/--normalize-box-size`로 통합),
> `scripts/tools_analyze_dataset.py`, `scripts/tools_visualize_predictions.py`

### 콜랩 `Untitled6.ipynb`가 실패한 이유

같은 시기 콜랩에서도 RetinaNet 이어학습을 시도했지만 마지막 셀이 에러로 끝났다.
코드를 보면 원인이 두 개 겹쳐 있다.

```python
self.label_mapping = {"aqua": 0, "string": 1, "floatingr": 2}
...
class_name = row['defect_class']
if class_name in self.label_mapping:
    labels.append(self.label_mapping[class_name])
else:
    raise ValueError(f"Unexpected class name '{defect_class}' found in data.")
```

1. CSV의 실제 `defect_class` 값은 `aq` / `st` / `fl`인데 매핑 키를
   `aqua` / `string` / `floatingr`로 적었다. 그래서 모든 행이 `else`로 빠진다.
2. 그 `else` 안의 에러 메시지가 참조하는 `defect_class`는 정의된 적이 없는
   이름이다(실제 변수명은 `class_name`). 그래서 의도한 `ValueError` 대신
   `NameError`가 난다.

원래 잡으려던 오류를 처리하다 새 오류를 낸 경우다. 아마 이 지점에서 콜랩을
포기하고 로컬로 넘어갔을 것이다.

> **정리본 대응:** 클래스 정의를 `src/seaweed/config.py` 한 곳으로 모으고,
> 알 수 없는 클래스가 들어오면 사용 가능한 값 목록과 함께 명확한 에러를 낸다.
> (`tests/test_data.py::test_unknown_defect_class_in_csv_raises`)

---

## 4단계 — 유형별 검출기 + 앙상블 (2024-11-18 ~ 11-23)

한 모델이 aq/st/fl 세 종류를 동시에 검출하고 분류하는 것이 잘 되지 않았다.
클래스별 데이터 수가 다르고(st 1,289 / aq 1,088 / fl 623) 결함 모양도 서로 달라,
한 모델의 용량을 세 문제에 나눠 쓰는 셈이었다.

그래서 **유형마다 "배경 vs 해당 결함" 2클래스 검출기를 따로 학습**하고, 추론 시
세 모델 중 가장 확신하는 예측 하나를 채택하는 구조로 바꿨다. 이미지당 결함이
하나뿐인 데이터 특성 덕분에 성립하는 단순한 규칙이다.

- `SeaWida_string/Retina_st_model.py` — 단일 클래스 검출기 학습
- `SeaWida_string/Retina_st_addtional_train.py` — 위와 같되 이어학습 지원
- `SeaWida_string_1123/model_ensemble.py` — 앙상블 평가 초기 버전
- `SeaWida_string_1123/ssshyun_is_the_best.py` — 앙상블 + 시각화
- `SeaWida_string_1123/dddddddddd.py` — **가장 발전된 최종 버전**
  (모델별 선택 횟수, 클래스별 정확도, IoU 미달 카운트 추가)
- `SeaWida_string_1123/counting_iou_low_75.py` — IoU 0.75 미달 사례 집계
- `SeaWida_string_1123/visual_bessssst_iou_hist.py`, `visual_less75_bbox_size.py` — 실패 분석

파일 이름과 실제 완성도가 반대인 경우가 있다. `ssshyun_is_the_best.py`보다
`dddddddddd.py`가 더 나중이고 더 완성된 버전이다. 마찬가지로
`SeaWida_string/Retina_st_model.py`는 파일명이 `st`인데 코드 안에서는
`defect_class == "fl"`을 거른다. 학습할 클래스를 소스 코드 문자열로 바꾸다 보니
파일명과 어긋난 채 저장된 것이다.

### 결과

`st` 검출기를 31에폭 학습한 기록이 남아 있다(28에폭 F1 0.7333이 최고).
전체 표와 해석은 [results/stage4_string_detector_metrics.md](results/stage4_string_detector_metrics.md).

당시 직접 남긴 `st_models_pick/README.txt`에는 31개 에폭 중 10개를 골라 보관한
이유가 적혀 있다("F1이 0.5를 처음 넘은 지점", "Precision과 Recall이 균형을 이루는
단계" 등). 체크포인트를 무작정 다 남기지 않고 근거를 적어 골라낸 기록이다.

> **정리본 대응:** `scripts/stage4_train_per_class.py`(`--defect-class`로 대상 지정),
> `scripts/stage4_eval_ensemble.py`, `src/seaweed/ensemble.py`

---

## 재검수에서 발견한 문제

정리하면서 원본 코드를 다시 읽고 실제로 실행해 확인한 것들이다.

### 1. 반전 증강에서 박스 좌표가 변환되지 않았다 (심각)

원본은 증강 변환을 리스트에 담고, 종류에 따라 박스도 함께 변환하려 했다.

```python
return [
    T.Compose([T.ToTensor(), T.RandomHorizontalFlip(p=1.0)]),        # 좌우 반전
    T.Compose([T.ToTensor(), T.RandomVerticalFlip(p=1.0)]),          # 상하 반전
    {"transform": T.Compose([...rotate(img, 90)...]), "angle": 90},  # 회전
]

def apply_transform_to_boxes(boxes, transform, w, h):
    if isinstance(transform, T.RandomHorizontalFlip):          # (1)
        ...
    elif isinstance(transform, T.RandomVerticalFlip):          # (2)
        ...
    elif isinstance(transform, dict) and "angle" in transform: # (3)
        ...
    return boxes
```

리스트에 들어 있는 것은 `RandomHorizontalFlip` 인스턴스가 아니라 그것을 감싼
`T.Compose` 객체다. 따라서 (1)과 (2)는 **항상 False**이고, 반전 증강에서는
박스가 변환되지 않은 채 그대로 반환된다. 회전은 dict 검사인 (3)에 걸려 정상
동작했다.

`Retina_st_model.py` 계열은 형태가 조금 다른데(전부 dict로 감쌌지만 반전
항목에는 `"angle"` 키가 없다) 결과는 같다 — 반전에서 박스가 안 움직인다.

실제로 원본 로직을 그대로 떼어 실행해 확인했다. 512×512 이미지의
`[100, 50, 111, 61]` 박스 기준:

| 변환 | 원본 코드 결과 | 정리본 결과 |
|------|----------------|-------------|
| 좌우 반전 | `[100, 50, 111, 61]` (그대로) | `[401, 50, 412, 61]` |
| 상하 반전 | `[100, 50, 111, 61]` (그대로) | `[100, 451, 111, 462]` |
| 90도 회전 | `[50, 401, 61, 412]` | `[50, 401, 61, 412]` (동일) |

**이미지는 뒤집혔는데 정답 박스는 원래 자리에 남은** 학습 샘플이 만들어졌다는
뜻이다. 증강 6종 중 2종이 여기 해당하므로 증강 데이터의 약 1/3이다. 결함이
512px 안에서 11px에 불과하니, 뒤집힌 좌표는 사실상 무작위 위치를 가리키는
오답 라벨이 된다. 모델 입장에서는 "아무것도 없는 곳에 결함이 있다"고 배우는
셈이라, **끝까지 Recall이 낮았던 유력한 원인 중 하나**다.

눈으로는 발견할 수 없는 종류의 버그다. 증강 이미지를 띄워 봐도 이미지 자체는
멀쩡하게 뒤집혀 있기 때문이다. 그래서 정리본에서는 변환 종류를 문자열 태그로
묶어 이미지와 박스가 어긋날 수 없게 만들고, 픽셀 단위로 확인하는 테스트를 붙였다
(`tests/test_transforms.py`).

### 2. 결함 클래스에 라벨 0을 할당했다

```python
self.label_mapping = {"aq": 0, "st": 1, "fl": 2}
```

torchvision 검출 모델은 라벨 0을 배경으로 예약한다. `aq` 결함이 배경과 같은
라벨을 갖게 되어 학습 신호가 뒤섞인다. 정리본은 1부터 시작한다
(`src/seaweed/config.py`).

### 3. 클래스 매핑이 파일마다 달랐다

- `Retina_addtional.py` — `{"aq": 0, "st": 1, "fl": 2}`
- `precision_recall_validate.py` — `{"st": 0, "aq": 1, "fl": 2}`
- `Untitled6.ipynb` — `{"aqua": 0, "string": 1, "floatingr": 2}`

앞의 둘은 `aq`와 `st`가 서로 뒤바뀐다. 같은 체크포인트를 다른 스크립트로
평가하면 클래스 정확도가 엉뚱하게 나온다. 세 번째는 CSV 실제 값과 아예 달라
실행 자체가 실패했다.

### 4. 앙상블 평가에서 검증셋 절반이 누락됐다

```python
dataloader = DataLoader(dataset, batch_size=2, ...)
for image_names, images, gt_boxes_list, gt_labels_list in dataloader:
    image = images[0].to(device)      # 배치의 두 번째 이미지는 버려진다
    gt_boxes = gt_boxes_list[0].numpy()
```

`batch_size=2`인데 배치의 첫 번째만 사용한다. 검증셋 750장 중 375장만
평가된 셈이다.

### 5. 시각화용 박스 확대가 지표를 오염시켰다

```python
def get_predictions(model, image, device, score_threshold=0.5):
    ...
    min_size = 10          # 너무 작아 안 보이니 키운다
    for box in boxes:
        if width < min_size: ...
    return boxes, scores, labels   # 이 박스가 그대로 IoU 계산으로 간다
```

정답 박스가 평균 11px인데 예측 박스를 최소 10px로 강제 확대하면 IoU가 크게
달라진다. 정리본은 확대를 시각화 스크립트에만 두고, 평가 경로는 원본 크기를 쓴다.

### 6. 평가 로직 자체가 두 갈래였다

`SeaWida_1116` 쪽 평가 함수는 예측 하나가 여러 정답과 겹칠 때 IoU를 중복
누적하고, 신뢰도 필터링 이전의 예측 수로 FP를 계산했다. `SeaWida_string` 쪽
평가 함수는 IoU 최대인 미매칭 정답을 고르는 그리디 매칭으로 제대로 되어 있다.
정리본은 후자를 채택하고 전자의 클래스 정확도 집계를 합쳤다
(`src/seaweed/metrics.py`).

### 7. `strict=False` 로딩이 실패를 숨겼다

```python
model.load_state_dict(torch.load(model_path), strict=False)
```

헤드를 교체한 뒤 형태가 안 맞는 키를 넘기려는 의도였겠지만, 이 설정에서는
**가중치가 하나도 안 불러와져도 에러 없이 통과**한다. 정리본은 strict 로딩을
먼저 시도하고 실패 시 무엇이 안 맞는지 출력한다.

### 8. 성능과 무관한 비효율

- 앙상블 평가에서 이미지 한 장 꺼낼 때마다 `pd.read_csv()`로 전체 CSV를 다시 읽었다.
- 일부 데이터셋이 CSV 행 단위로 인덱싱해서, 박스가 여러 개인 이미지를 중복 로드했다.

---

## 지금 다시 한다면

1. **버그를 고친 상태로 재학습부터.** `results/`의 수치는 전부 증강 라벨이
   오염된 상태에서 나온 것이다. 정상 증강으로 다시 돌리면 Recall이 어디까지
   오르는지가 이 프로젝트의 남은 질문이다.
2. **무결함 이미지를 검출 평가에 포함.** 지금 검증셋은 결함 이미지 750장뿐이라
   "정상 김을 결함이라고 오탐하는 비율"을 측정하지 못한다. 실제 공정에서는 이
   지표가 더 중요하다.
3. **작은 객체 검출 정석 적용.** 512px 이미지를 타일로 잘라 확대해 학습하거나,
   FPN의 더 얕은 층까지 앵커를 배치하는 방식. 11px 객체를 원본 해상도에서
   그대로 잡으려 한 것이 근본적인 어려움이었다.
4. **에폭 선택을 검증셋에 의존하지 않기.** 유형당 150~310장으로는 에폭 간
   F1 변동이 커서, 최고 F1 에폭 고르기가 검증셋 과적합이 된다.

---

## 원본 파일 대응표

45개 원본 파일이 각각 어디로 갔는지는 [docs/original_file_map.md](docs/original_file_map.md)에 있다.
