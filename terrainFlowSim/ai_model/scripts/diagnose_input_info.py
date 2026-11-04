"""
입력(등고선 그림)에 배수 물길 정보가 남아 있는가?

기울기 손실을 넣어도 번짐(sigma≈2.2)이 전혀 줄지 않았다. 손실로 "날카롭게 그려라"라고
요구해도 안 된다면, 모델이 못 하는 게 아니라 **입력에 그 정보가 없을** 가능성이 있다.

의심 지점: 학습 입력은 `contourf(levels=20)` 로 그린 등고선이다. 이건 고도를 20단계로
**양자화**한다. 고저차가 25m면 한 단계가 1.25m인데, 얇은 배수 물길이 파인 깊이는 보통
그보다 훨씬 얕다. 그렇다면 물길의 위치가 입력 그림에 애초에 담기지 않는다.

검증 방법: 지형에서 D8 흐름누적(물이 모이는 경로)을 계산하되,
  (a) 원본 고도
  (b) 20단계로 양자화한 고도   <- 등고선 그림이 담고 있는 정보량
둘에서 각각 구해 비교한다. (b)에서 물길이 무너진다면 입력이 병목이다.

정답 수심과의 상관도 같이 본다. "정답을 설명하는 능력"이 양자화로 얼마나 깎이는지가
모델이 넘을 수 없는 입력 쪽 천장이다.

사용법:
    .venv\\Scripts\\python.exe scripts\\diagnose_input_info.py --samples 40
"""

import os
import sys
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.utils.flow_accum import d8_flow_accumulation  # noqa: E402
from src.ground_truth import GroundTruth, REGIONS  # noqa: E402
from src.datasets.contour_flow_dataset import split_train_val  # noqa: E402


def quantize(z, levels):
    """contourf(levels=N) 이 담는 정보량 흉내: 고도를 N개 구간으로 나눠 구간 대표값으로."""
    lo, hi = float(z.min()), float(z.max())
    if hi - lo < 1e-9:
        return z.copy()
    edges = np.linspace(lo, hi, levels + 1)
    idx = np.clip(np.digitize(z, edges) - 1, 0, levels - 1)
    mids = (edges[:-1] + edges[1:]) * 0.5
    return mids[idx]


def pearson(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--levels", type=int, default=20, help="등고선 단계 수 (학습 렌더와 동일)")
    ap.add_argument("--min-depth", type=float, default=0.05)
    args = ap.parse_args()

    _, val_names = split_train_val(os.path.join(ROOT, "data", "contours"),
                                   os.path.join(ROOT, "data", "flow"), 0.15, 42)
    names = sorted(n for n in val_names if "_r0_t" in n)

    gt_src = GroundTruth()
    rows = []
    for n in names:
        if len(rows) >= args.samples:
            break
        g = gt_src.get(n)
        if g is None or g["depth_m"].max() < args.min_depth:
            continue
        z = np.load(os.path.join(REGIONS[g["region"]], g["sample"], "terrain.npy"))
        if z.shape != g["depth_m"].shape:
            continue
        depth = g["depth_m"]

        zq = quantize(z, args.levels)
        band = (float(z.max()) - float(z.min())) / args.levels

        fa = np.log1p(d8_flow_accumulation(z))
        faq = np.log1p(d8_flow_accumulation(zq))

        rows.append({
            "relief": float(np.ptp(z)),
            "band": band,
            "fa_vs_faq": pearson(fa, faq),
            "depth_vs_fa": pearson(depth, fa),
            "depth_vs_faq": pearson(depth, faq),
            # 물길(상위 5% 흐름누적)이 양자화 후에도 같은 자리에 남는가
            "chan_iou": channel_iou(fa, faq, 95),
        })

    if not rows:
        print("표본 없음")
        return

    m = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    print(f"표본 {len(rows)}개 (val, 정답 최대수심 {args.min_depth}m 이상)\n")
    print(f"평균 고저차 {m['relief']:.1f} m  ->  등고선 {args.levels}단계면 한 칸 "
          f"**{m['band']:.2f} m**\n")

    print("## 양자화가 물길에 미치는 영향\n")
    print("| 항목 | 값 |")
    print("|---|---:|")
    print(f"| 흐름누적 상관 (원본 vs 20단계) | {m['fa_vs_faq']:.3f} |")
    print(f"| 물길 위치 일치 IoU (상위 5%) | {m['chan_iou']:.3f} |")
    print()
    print("## 정답 수심을 설명하는 능력\n")
    print("| 입력 | 정답 수심과의 상관 |")
    print("|---|---:|")
    print(f"| 원본 고도의 흐름누적 | {m['depth_vs_fa']:.3f} |")
    print(f"| 20단계 양자화 고도의 흐름누적 | {m['depth_vs_faq']:.3f} |")
    drop = m["depth_vs_fa"] - m["depth_vs_faq"]
    print(f"\n양자화로 잃는 설명력: **{drop:+.3f}** "
          f"({drop / max(m['depth_vs_fa'], 1e-9) * 100:.0f}%)")

    print("\n## 판정\n")
    if m["chan_iou"] < 0.5 or drop > 0.05:
        print("양자화가 물길 정보를 크게 깎는다.")
        print("-> **입력이 병목**이다. 손실이나 아키텍처로는 넘을 수 없다.")
        print("   처방: 등고선 그림 대신 **고도 격자를 그대로 입력**으로 쓴다.")
    else:
        print("양자화 후에도 물길이 대체로 살아 있다.")
        print("-> 입력은 병목이 아니다. 번짐의 원인은 다른 곳(아키텍처/해상도)에 있다.")


def channel_iou(fa, faq, pct):
    a = fa >= np.percentile(fa, pct)
    b = faq >= np.percentile(faq, pct)
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else float("nan")


if __name__ == "__main__":
    main()
