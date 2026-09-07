# 김 이물질(해조류 결함) 검출

512×512 김 이미지에서 이물질을 찾아내는 딥러닝 모델.
**[2024 DATA·AI 분석 경진대회](https://aida.kisti.re.kr/competition/main/problem/PROB_000000000000392/detail.do)**
(한국과학기술정보연구원 KISTI 주최) 모델 개발 부문에 참가하며 진행한 프로젝트로,
**이진분류 → 객체탐지 → 결함 유형별 앙상블** 순으로 접근을 네 번 바꿔가며 작업했다.
서울여자대학교 데이터분석 소학회 WIDA의 2024-2학기 활동으로 함께 진행했다.

이 저장소는 그 실험 과정을 정리한 것이다. 당시 코드는 콜랩 노트북 3개와 로컬
PyCharm 프로젝트 5개에 걸쳐 파이썬 파일 45개(약 7,800줄)로 흩어져 있었고,
대부분은 같은 스크립트를 조금씩 고쳐 복사한 것이었다. 그 45개를 실행 순서와
역할에 따라 재구성하고, 정리 과정에서 발견한 버그를 고쳤다.

> 학부 2학년 때의 프로젝트이며, 실무 배포용이 아니라 **실험 기록**이다.
> 무엇을 왜 시도했고 어디서 막혔는지를 남기는 데 무게를 뒀다.
> 정리하며 발견한 원본의 문제점도 숨기지 않고 [EXPERIMENTS.md](EXPERIMENTS.md)에 적었다.

---

## 대회 과제

> **김(Seaweed) 이미지에서 이물질 결함 검출 및 품질 분류 AI 모델 개발**
>
> 김은 최근 중요한 수출 품목으로 대두되고 있으나 품질 관리가 수출 경쟁력 유지에
> 핵심적인 요소다. 기존 김 품질 검사는 육안 검사에 의존해 시간과 비용이 많이 들고
> 정확도가 낮아, 자동화된 결함 검출·품질 분류 시스템이 필요하다.
>
> 요구사항:
> 1. 김의 품질에 따라 등급을 분류 (결함 / 무결함)
> 2. 이물질 결함을 자동으로 검출
> 3. **결함의 위치와 유형을 정확히 식별**

이 저장소의 4단계 구성은 이 세 요구사항을 순서대로 따라간 결과다. 1단계
이진분류가 요구사항 1(품질 등급 분류)에, 2~4단계 객체탐지가 요구사항 2·3(결함의
위치와 유형 식별)에 대응한다. 특히 요구사항 3의 "유형까지" 때문에 결함을 단순히
찾는 데서 그치지 않고 `aq`/`st`/`fl` 세 유형을 구분해야 했고, 그것이 4단계의
유형별 검출기 + 앙상블 구조로 이어졌다.

- 대회: 2024 DATA·AI 분석 경진대회 (KISTI), 참가신청 2024-08-23 ~ 2024-10-25
- 진행 시기: 2024년 9월 ~ 11월
- 데이터: 대회에서 제공한 김 이미지 데이터셋 ([AIDA 플랫폼](https://aida.kisti.re.kr))

---

## 문제 정의

김 생산 공정에서 완성된 김에 섞여 들어간 이물질을 자동으로 찾아내는 것이 목표다.
결함은 세 종류로 라벨링되어 있다.

| 코드 | 의미 | 학습셋 | 검증셋 |
|------|------|-------:|-------:|
| `aq` | aqua — 해수 얼룩 | 1,088 | 287 |
| `st` | string — 실 모양 이물질 | 1,289 | 312 |
| `fl` | floating — 부유물 | 623 | 151 |
| | **합계** | **3,000** | **750** |

### 이 문제가 어려운 이유

**결함이 매우 작다.** 512×512 이미지 안에서 결함 박스는 평균 11×11px, 최소 3px다.
이미지 면적의 약 0.05%에 불과하다. 사전학습 검출 모델의 기본 앵커 크기(32~512px)로는
아예 후보로도 잡히지 않아서, 앵커 크기를 내리는 것부터 시작해야 했다.

**클래스 불균형이 극심하다.** 원본 데이터는 결함 이미지 3,000장에 무결함 이미지
255,000장이다(약 1:85). 아무것도 검출하지 않고 전부 "정상"이라고 답해도 정확도가
99% 넘게 나오므로, 정확도라는 지표 자체가 무의미하다. 1단계에서 한참을 헤맨 이유가
여기에 있다.

---

## 접근의 변천

```
1단계  결함 유무만 판단          ResNet18/50 이진분류        <- 요구사항 1 (품질 등급 분류)
  ↓    위치·유형까지 요구된다
2단계  결함 위치 검출            Faster R-CNN (앵커 축소)     <- 요구사항 2 (결함 검출)
  ↓    작은 객체에 더 강한 모델로
3단계  RetinaNet + 데이터 증강    RetinaNet(ResNet50-FPN), 반전·회전 증강, 이어학습
  ↓    한 모델이 3종을 다 잘 잡지 못한다
4단계  유형별 검출기 + 앙상블     aq/st/fl 각각 2클래스 검출기 <- 요구사항 3 (위치 + 유형)
                                → 신뢰도 최댓값 선택
```

각 단계의 상세한 배경과 실패 기록은 [EXPERIMENTS.md](EXPERIMENTS.md)에 있다.

## 결과

4단계 `st` 검출기 기준 최고 성능(31에폭 학습, IoU 0.75 / 신뢰도 0.5 기준):

| Precision | Recall | F1 | Average IoU |
|----------:|-------:|-----:|------------:|
| 0.8684 | 0.6346 | **0.7333** | 0.8732 |

에폭별 전체 기록과 해석은 [results/stage4_string_detector_metrics.md](results/stage4_string_detector_metrics.md).

핵심은 **Precision은 초반부터 0.8대인데 Recall이 0.05에서 시작해 0.63까지
겨우 올라간다**는 점이다. 위치를 잡기만 하면 IoU는 0.87로 정확했다. 즉 이
문제의 난이도는 "정확히 어디인가"가 아니라 **"있다는 것을 알아채는가"**에 있었다.

---

## 저장소 구조

```
seaweed-defect-detection/
├── src/seaweed/              공용 모듈
│   ├── config.py             클래스 정의, 상수, 기본 경로
│   ├── data.py               데이터셋 3종 (분류 / 다중클래스 검출 / 단일클래스 검출)
│   ├── transforms.py         박스 좌표까지 함께 변환하는 증강
│   ├── models.py             ResNet 분류기 / Faster R-CNN / RetinaNet 생성
│   ├── metrics.py            IoU, 검출 지표 집계
│   └── ensemble.py           4단계 앙상블
├── scripts/                  실행 스크립트 (전부 argparse CLI)
│   ├── stage1_train_classifier.py
│   ├── stage2_train_faster_rcnn.py
│   ├── stage3_train_retinanet.py
│   ├── stage4_train_per_class.py
│   ├── stage4_eval_ensemble.py
│   ├── tools_json_to_csv.py
│   ├── tools_analyze_dataset.py
│   └── tools_visualize_predictions.py
├── tests/                    pytest (36개)
├── results/                  당시 학습 기록
├── EXPERIMENTS.md            실험 연대기 + 원본 파일 매핑 + 발견한 버그
└── docs/original_file_map.md 원본 45개 파일이 각각 어디로 갔는지
```

## 설치

```bash
git clone <this-repo>
cd seaweed-defect-detection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

GPU 학습은 CUDA 버전에 맞는 PyTorch가 필요하다
([공식 설치 안내](https://pytorch.org/get-started/locally/)).

## 데이터 준비

**데이터셋은 이 저장소에 포함되어 있지 않다.** 2024 DATA·AI 분석 경진대회에서
참가자에게 제공된 데이터이므로 재배포하지 않으며(용량도 무결함 이미지만 25만 장이다),
데이터가 필요하면 [KISTI AIDA 플랫폼](https://aida.kisti.re.kr)에서 확인해야 한다.

코드에 하드코딩된 경로는 없다. 데이터를 어디에 두든 아래 구조만 지키고 각
스크립트에 경로를 인자로 넘기면 된다.

```
data/
├── train/
│   ├── train_defected_dataset/     결함 이미지 (*.png)
│   ├── train_defected_json/        이미지당 라벨 JSON 1개
│   └── train_pure_dataset/         무결함 이미지
├── validation/
│   ├── validation_defected_dataset/
│   ├── validation_defected_json/
│   └── validation_pure_dataset/
├── filtered_seaweed.csv            학습 라벨 (JSON을 합친 것)
└── val_seaweed.csv                 검증 라벨
```

JSON 라벨은 이미지 한 장당 한 파일이다.

```json
{"image_name": "seaweed_01250.png", "defect_class": "st",
 "top_x": 87, "top_y": 104, "bot_x": 100, "bot_y": 116}
```

CSV로 합치기:

```bash
python scripts/tools_json_to_csv.py \
    --json-dir data/train/train_defected_json \
    --output   data/filtered_seaweed.csv
```

데이터 특성부터 확인하고 싶다면:

```bash
python scripts/tools_analyze_dataset.py --csv data/filtered_seaweed.csv \
    --output-dir outputs/dataset_stats
```

## 실행

### 1단계 — 이진분류

```bash
python scripts/stage1_train_classifier.py \
    --train-defected data/train/train_defected_dataset \
    --train-pure     data/train/train_pure_dataset \
    --val-defected   data/validation/validation_defected_dataset \
    --val-pure       data/validation/validation_pure_dataset \
    --max-pure 3000 --epochs 10 --pos-weight 2.0
```

`--max-pure`로 무결함 이미지 수를 제한하는 것이 중요하다. 25만 장을 그대로
넣으면 모델이 전부 "정상"으로 찍는 쪽으로 수렴한다.

### 2단계 — Faster R-CNN

```bash
python scripts/stage2_train_faster_rcnn.py \
    --train-images data/train/train_defected_dataset \
    --train-csv    data/filtered_seaweed.csv \
    --val-images   data/validation/validation_defected_dataset \
    --val-csv      data/val_seaweed.csv \
    --epochs 7 --augment
```

### 3단계 — RetinaNet

```bash
python scripts/stage3_train_retinanet.py \
    --train-images data/train/train_defected_dataset \
    --train-csv    data/filtered_seaweed.csv \
    --val-images   data/validation/validation_defected_dataset \
    --val-csv      data/val_seaweed.csv \
    --augment --epochs 20 --output-dir outputs/stage3

# 중단한 지점부터 이어학습
python scripts/stage3_train_retinanet.py ... \
    --resume outputs/stage3/retinanet_epoch_20.pth --start-epoch 21 --epochs 10

# 박스 정규화 실험 (모든 정답 박스를 중심 유지 16px 정사각형으로)
python scripts/stage3_train_retinanet.py ... --normalize-box-size 16
```

### 4단계 — 유형별 검출기 + 앙상블

```bash
# 결함 유형별로 따로 학습
for C in aq st fl; do
  python scripts/stage4_train_per_class.py --defect-class $C \
      --train-images data/train/train_defected_dataset \
      --train-csv    data/filtered_seaweed.csv \
      --val-images   data/validation/validation_defected_dataset \
      --val-csv      data/val_seaweed.csv \
      --augment --epochs 30 --output-dir outputs/stage4/$C
done

# 셋을 묶어 평가
python scripts/stage4_eval_ensemble.py \
    --val-images data/validation/validation_defected_dataset \
    --val-csv    data/val_seaweed.csv \
    --checkpoint aq=outputs/stage4/aq/retinanet_aq_epoch_28.pth \
    --checkpoint st=outputs/stage4/st/retinanet_st_epoch_28.pth \
    --checkpoint fl=outputs/stage4/fl/retinanet_fl_epoch_30.pth \
    --save-ensemble outputs/stage4/ensemble.pth

# 틀린 사례만 뽑아 시각화 (오답 분석)
python scripts/tools_visualize_predictions.py \
    --val-images data/validation/validation_defected_dataset \
    --val-csv    data/val_seaweed.csv \
    --checkpoint aq=... --checkpoint st=... --checkpoint fl=... \
    --only-failures --num-images 20 --output-dir outputs/vis
```

## 테스트

```bash
pytest tests/ -v          # 단위 테스트 36개 (수 초)
bash tests/smoke_test.sh  # 전체 파이프라인 스모크 테스트 (CPU 8~10분)
```

`smoke_test.sh`는 합성 데이터를 만들어 8개 스크립트를 전부 1~2에폭씩 실행한다.
성능을 보려는 게 아니라 모든 실행 경로가 살아 있는지 확인하는 용도다.
사전학습 가중치를 받을 수 없는 환경을 위해 모든 학습 스크립트에 `--no-pretrained`
옵션이 있다.

증강 검증 테스트가 핵심이다. 박스 영역만 흰색으로 칠한 합성 이미지를 만들어,
변환 후에도 변환된 박스 좌표가 흰 영역을 픽셀 단위로 정확히 덮는지 확인한다.
원본에 있던 "이미지만 뒤집히고 박스는 제자리에 남는" 버그는 눈으로는 보이지
않기 때문에, 이런 형태의 테스트가 없으면 학습 데이터가 조용히 오염된다.

## 정리하며 고친 것

자세한 내용은 [EXPERIMENTS.md](EXPERIMENTS.md)의 "재검수에서 발견한 문제"에 있다.
요약하면:

1. **반전 증강에서 박스 좌표가 변환되지 않았다** — 이미지는 뒤집혔는데 정답
   박스는 원래 자리에 남아, 증강 데이터의 상당 부분이 오답 라벨이었다.
   Recall이 끝까지 낮았던 유력한 원인 중 하나다.
2. **결함 클래스에 라벨 0을 할당해 배경과 충돌했다** — torchvision 검출 모델은
   라벨 0을 배경으로 예약한다.
3. **클래스 매핑이 파일마다 달랐다** (`{aq:0,st:1,fl:2}` vs `{st:0,aq:1,fl:2}` vs
   `{aqua:0,string:1,floatingr:2}`) — 마지막 것은 CSV 실제 값과 아예 달라 콜랩
   실행이 실패한 원인이었다.
4. **앙상블 평가에서 검증셋 절반이 조용히 누락됐다** — `batch_size=2`로 만들고
   `images[0]`만 썼다.
5. **시각화용 박스 확대가 IoU 계산까지 오염시켰다** — 정답 박스가 11px인데
   예측 박스를 최소 10px로 강제 확대한 뒤 그대로 IoU를 쟀다.

## 알려진 한계

- 검증셋이 유형당 150~310장 수준이라 에폭 간 F1 변동이 크다. 최고 F1 에폭을
  고르는 방식 자체가 검증셋에 과적합일 수 있다.
- 앙상블이 "이미지당 결함 1개"를 전제로 신뢰도 최댓값 하나만 남긴다. 결함이
  여러 개인 이미지에는 그대로 쓸 수 없다.
- 무결함 이미지는 4단계 검출 학습에 전혀 쓰이지 않았다. 실제 공정에서는
  정상 김을 결함으로 오탐하지 않는 것이 중요한데, 그 평가가 빠져 있다.
- 위 버그들을 고친 뒤 재학습한 결과는 아직 없다. `results/`의 수치는 모두
  **버그가 있는 상태로 학습·측정된 값**이다.

## 라이선스

MIT — [LICENSE](LICENSE) 참고. 데이터셋은 포함되어 있지 않다.
