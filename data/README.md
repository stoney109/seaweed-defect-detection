# 데이터 폴더

데이터셋은 저장소에 포함하지 않는다. 2024 DATA·AI 분석 경진대회(KISTI)에서
참가자에게 제공된 데이터라 재배포하지 않으며, 용량도 무결함 이미지만 25만 장이다.
데이터가 필요하면 [KISTI AIDA 플랫폼](https://aida.kisti.re.kr)에서 확인할 것.

코드에는 하드코딩된 경로가 없다. 데이터를 어디에 두든 아래 구조만 지키고
스크립트에 경로를 인자로 넘기면 된다.

```
data/
├── train/
│   ├── train_defected_dataset/       결함 이미지 (*.png, 512x512)
│   ├── train_defected_json/          이미지당 라벨 JSON 1개
│   └── train_pure_dataset/           무결함 이미지
├── validation/
│   ├── validation_defected_dataset/
│   ├── validation_defected_json/
│   └── validation_pure_dataset/
├── filtered_seaweed.csv              학습 라벨
└── val_seaweed.csv                   검증 라벨
```

## 라벨 형식

JSON (이미지 한 장당 한 파일):

```json
{"image_name": "seaweed_01250.png", "defect_class": "st",
 "top_x": 87, "top_y": 104, "bot_x": 100, "bot_y": 116}
```

CSV (`scripts/tools_json_to_csv.py`로 생성):

```
image_name,defect_class,top_x,top_y,bot_x,bot_y,label
seaweed_01423.png,fl,206.0,266.0,218.0,278.0,1
```

`defect_class`는 `aq`(해수 얼룩) / `st`(실 모양) / `fl`(부유물) 세 가지다.

## 참고 통계

| 항목 | 학습 | 검증 |
|------|-----:|-----:|
| 결함 이미지 | 3,000 | 750 |
| `aq` / `st` / `fl` | 1,088 / 1,289 / 623 | 287 / 312 / 151 |
| 무결함 이미지 (원본) | 255,000 | 40,294 |

결함 박스 크기: 평균 11x11px, 최소 3px, 최대 21px (512x512 이미지 기준).
