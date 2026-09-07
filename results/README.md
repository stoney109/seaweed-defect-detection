# 당시 학습 기록

2024년 11월 프로젝트 진행 당시 남긴 실제 기록이다. 정리하면서 새로 만든 것이
아니라 원본을 옮겨온 것이므로, **수치는 모두 버그가 있는 상태로 학습·측정된 값**이다
(자세한 내용은 [../EXPERIMENTS.md](../EXPERIMENTS.md)의 "재검수에서 발견한 문제").

| 파일 | 내용 |
|------|------|
| `stage4_string_detector_metrics.md` | `st` 검출기 31에폭 학습 기록을 표로 정리하고 해석을 붙인 것 |
| `st_model_validation_original.txt` | 위의 원본. 당시 학습 로그에서 손으로 옮겨 적은 텍스트 |
| `checkpoint_selection_original.txt` | 31개 에폭 중 10개를 골라 보관한 이유를 직접 적어둔 메모 |

`checkpoint_selection_original.txt`는 체크포인트를 무작정 전부 남기지 않고
"F1이 0.5를 처음 넘은 지점", "Precision과 Recall이 균형을 이루는 단계"처럼
근거를 적어 골라낸 기록이다. 당시 실험 관리 방식을 보여주는 자료라 그대로 남겼다.

## 정리본으로 다시 측정하려면

체크포인트 파일(`.pth`)은 용량 때문에 저장소에 포함하지 않았다. 가지고 있는
체크포인트로 다시 재려면:

```bash
python scripts/stage4_eval_ensemble.py \
    --val-images data/validation/validation_defected_dataset \
    --val-csv    data/val_seaweed.csv \
    --checkpoint aq=<경로> --checkpoint st=<경로> --checkpoint fl=<경로>
```

평가 로직을 고쳤기 때문에 같은 체크포인트라도 위 표와 수치가 달라진다.
