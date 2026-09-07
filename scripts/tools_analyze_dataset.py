"""데이터셋 통계 분석 — 결함 박스 크기 분포, 클래스 분포.

이 프로젝트에서 검출이 유독 어려웠던 이유가 여기서 드러난다. 512x512 이미지
안에서 결함 박스가 평균 11x11px(면적 기준 약 0.05%)에 불과하다. 기본 앵커
크기로는 잡히지 않는 크기라서 3~4단계에서 앵커와 IoU 기준을 계속 조정했다.

원본 대응 파일: `SeaWida/visualization_width_height.py`, `SeaWida/ddd..py`,
`SeaWida_string_1123/visual_less75_bbox_size.py`
(같은 히스토그램 코드가 폴더마다 복사되어 있었다).

실행 예:

    python scripts/tools_analyze_dataset.py --csv data/filtered_seaweed.csv \\
        --output-dir outputs/dataset_stats
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="결함 박스 크기·클래스 분포 분석")
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=None, help="히스토그램 저장 폴더")
    p.add_argument("--bins", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    df["width"] = df["bot_x"] - df["top_x"]
    df["height"] = df["bot_y"] - df["top_y"]
    df["area"] = df["width"] * df["height"]

    print(f"[파일] {args.csv}")
    print(f"[행 수] 박스 {len(df)}개 / 이미지 {df['image_name'].nunique()}장")
    print(f"[클래스 분포] {df['defect_class'].value_counts().to_dict()}")
    print("\n[박스 크기 통계]")
    print(df[["width", "height", "area"]].describe().round(2).to_string())

    print("\n[클래스별 평균 박스 크기]")
    print(
        df.groupby("defect_class")[["width", "height", "area"]]
        .mean()
        .round(2)
        .to_string()
    )

    if args.output_dir is None:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")  # 화면 없는 환경에서도 저장 가능하도록
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib이 없어 히스토그램은 건너뜁니다. (pip install matplotlib)")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for column, title in [
        ("width", "Defect Width Distribution"),
        ("height", "Defect Height Distribution"),
        ("area", "Defect Area Distribution"),
    ]:
        plt.figure(figsize=(8, 6))
        plt.hist(df[column], bins=args.bins, alpha=0.75, edgecolor="black")
        plt.title(title)
        plt.xlabel(f"{column} (pixels)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        out = args.output_dir / f"{column}_hist.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"저장: {out}")

    plt.figure(figsize=(8, 6))
    for defect_class, group in df.groupby("defect_class"):
        plt.scatter(group["width"], group["height"], alpha=0.5, label=defect_class)
    plt.title("Width vs Height by Defect Class")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = args.output_dir / "width_vs_height.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
