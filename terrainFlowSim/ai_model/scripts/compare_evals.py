"""
evaluate.py 가 만든 여러 결과(eval_<tag>/)를 한 표로 비교한다.

summary.md 의 '전체' 행만으로는 부족하다. 이 과제의 지배적 오차인 **최대 수심 비율**
(피크를 얼마나 살렸는가)은 per_sample.csv 에서 다시 계산해야 하고, 그게 Phase 2
목적함수 실험의 승패를 가르는 지표다.

사용법:
    .venv\\Scripts\\python.exe scripts\\compare_evals.py v5 p2_base p2_wet8
"""

import os
import sys
import csv
import argparse

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def load(tag):
    path = os.path.join(ROOT, f"eval_{tag}", "per_sample.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    if not rows:
        return None

    def col(k):
        return np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])

    px, gx = col("pred_max_m"), col("gt_max_m")
    return {
        "tag": tag,
        "n": len(rows),
        "mae_m": np.nanmean(col("mae_m")),
        "r": np.nanmean(col("r")),
        "iou_1cm": np.nanmean(col("iou_1cm")),
        "iou_5cm": np.nanmean(col("iou_5cm")),
        "peak_err_m": np.nanmean(col("peak_err_m")),
        "peak_ratio": np.nanmedian(px / np.maximum(gx, 1e-9)),
        "bias_m": np.nanmean(col("bias_m")),
        "regions": {reg: np.nanmean([float(r["r"]) for r in rows if r["region"] == reg])
                    for reg in sorted({r["region"] for r in rows})},
    }


COLS = [("mae_m", "MAE[m]", "{:.4f}", -1), ("r", "r", "{:.4f}", +1),
        ("iou_1cm", "IoU@1cm", "{:.4f}", +1), ("iou_5cm", "IoU@5cm", "{:.4f}", +1),
        ("peak_err_m", "최심점오차[m]", "{:.1f}", -1),
        ("peak_ratio", "최대수심 비율", "{:.3f}", 0), ("bias_m", "편향[m]", "{:+.5f}", 0)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tags", nargs="+")
    args = p.parse_args()

    res = [r for r in (load(t) for t in args.tags) if r]
    missing = [t for t in args.tags if t not in {r["tag"] for r in res}]
    if missing:
        print(f"결과 없음(건너뜀): {', '.join(missing)}")
    if not res:
        return

    print("\n| 지표 | " + " | ".join(r["tag"] for r in res) + " |")
    print("|---|" + "---:|" * len(res))
    print("| 샘플 수 | " + " | ".join(str(r["n"]) for r in res) + " |")
    for key, label, fmt, better in COLS:
        vals = [r[key] for r in res]
        cells = [fmt.format(v) for v in vals]
        if better and len(vals) > 1:
            best = int(np.argmax(vals) if better > 0 else np.argmin(vals))
            cells[best] = f"**{cells[best]}**"
        print(f"| {label} | " + " | ".join(cells) + " |")

    regions = sorted({reg for r in res for reg in r["regions"]})
    if regions:
        print("\n지역별 상관계수 r\n")
        print("| 지역 | " + " | ".join(r["tag"] for r in res) + " |")
        print("|---|" + "---:|" * len(res))
        for reg in regions:
            print(f"| {reg} | " + " | ".join(f"{r['regions'].get(reg, float('nan')):.4f}"
                                             for r in res) + " |")

    print("\n최대수심 비율은 1.000 이 목표(파이프라인 천장 0.93). "
          "굵은 글씨는 각 행에서 가장 좋은 값.")


if __name__ == "__main__":
    main()
