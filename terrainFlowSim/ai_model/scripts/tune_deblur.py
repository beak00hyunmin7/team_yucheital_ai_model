"""
측정된 흐림을 후처리로 되돌린다 (학습 불필요).

근거: scripts/diagnose_sharpness.py 로 이 모델의 출력이 정답을 sigma≈2.2 로 흐린 것과
거의 같다는 것을 측정했다. 두 추정(예측과 가장 가까워지는 sigma, 피크 비율을 설명하는
sigma)이 일치했고, 십분위 표도 "얕은 구간 과대 + 가장 깊은 구간 과소"라는 번짐의 지문을
보였다. 흐림 커널을 알면 역연산을 걸 수 있다.

방법: 언샤프 마스킹.  sharp = pred + amount * (pred - blur(pred, sigma))
가우시안 역합성곱(Wiener/RL)보다 거칠지만 잡음 증폭에 훨씬 덜 취약하다.

**중요 - 튜닝과 보고를 분리한다.** 파라미터(sigma, amount)는 **train split** 에서 고르고,
성능은 **val split** 에서 잰다. 같은 데이터에서 고르고 보고하면 그건 성능이 아니라
그 데이터에 맞춘 정도다.

사용법:
    .venv\\Scripts\\python.exe scripts\\tune_deblur.py --checkpoint checkpoints_p2_wet8/best.pt \
        --config configs/config_p2_wet8.yaml
"""

import os
import sys
import json
import argparse

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.datasets.contour_flow_dataset import split_train_val  # noqa: E402
from src.ground_truth import GroundTruth, to_array_orientation, compare  # noqa: E402
from evaluate import Predictor  # noqa: E402

SIGMAS = [1.0, 1.5, 2.0, 2.5, 3.0]
AMOUNTS = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0]
VMAX = 0.6


def unsharp(x, sigma, amount):
    if amount <= 0:
        return x
    return np.clip(x + amount * (x - gaussian_filter(x, sigma)), 0.0, VMAX)


def collect(pred, gt_src, contours_dir, rain, tval, names, limit, min_depth):
    """(예측 수심, 정답 수심) 쌍을 모은다."""
    out = []
    for n in names:
        if len(out) >= limit:
            break
        g = gt_src.get(n)
        if g is None or g["depth_m"].max() < min_depth:
            continue
        p = to_array_orientation(
            pred.to_depth(pred.forward([os.path.join(contours_dir, n)],
                                       [pred.cond(rain.get(n), tval.get(n))]),
                          g["depth_m"].shape)[0])
        out.append((p, g["depth_m"]))
    return out


def score(pairs, sigma, amount):
    """후처리 후 지표 평균. 판정 기준은 이 과제의 목적에 가까운 IoU@5cm 와 피크 비율."""
    acc = {"mae_m": [], "r": [], "iou_1cm": [], "iou_5cm": [], "peak_ratio": []}
    for p, g in pairs:
        m = compare(unsharp(p, sigma, amount), g)
        for k in acc:
            v = m.get(k)
            if v is not None and np.isfinite(v):
                acc[k].append(v)
    return {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_p2_wet8/best.pt")
    ap.add_argument("--config", default="configs/config_p2_wet8.yaml")
    ap.add_argument("--tune-samples", type=int, default=50)
    ap.add_argument("--test-samples", type=int, default=120)
    ap.add_argument("--min-depth", type=float, default=0.05)
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    contours_dir = os.path.join(ROOT, cfg["contours_dir"])
    flow_dir = os.path.join(ROOT, cfg["flow_dir"])
    train_names, val_names = split_train_val(contours_dir, flow_dir,
                                             val_split=cfg["val_split"], seed=cfg["seed"])
    tr = sorted(n for n in train_names if "_r0_t" in n)
    va = sorted(n for n in val_names if "_r0_t" in n)

    rain = json.load(open(os.path.join(ROOT, cfg["rain_values_json"]), encoding="utf-8"))
    tval = json.load(open(os.path.join(ROOT, cfg["time_values_json"]), encoding="utf-8"))

    elev_dir = cfg.get("elev_dir")
    pred = Predictor(os.path.join(ROOT, args.checkpoint), cfg["image_size"],
                     elev_dir=os.path.join(ROOT, elev_dir) if elev_dir else None,
                     elev_only=bool(cfg.get("elev_only")))
    gt_src = GroundTruth()

    print(f"체크포인트 {args.checkpoint}")
    print("튜닝: train split / 보고: val split (같은 데이터에서 고르고 보고하지 않는다)\n")

    tune = collect(pred, gt_src, contours_dir, rain, tval, tr, args.tune_samples, args.min_depth)
    print(f"튜닝 표본 {len(tune)}개")

    best, rows = None, []
    for s in SIGMAS:
        for a in AMOUNTS:
            m = score(tune, s, a)
            rows.append((s, a, m))
            # 목적에 가까운 지표로 고른다: IoU@5cm 우선, 동률이면 피크 비율
            key = (m["iou_5cm"], m["peak_ratio"])
            if best is None or key > best[0]:
                best = (key, s, a, m)
    _, bs, ba, bm = best
    print(f"\n최적: sigma={bs}, amount={ba}  (IoU@5cm {bm['iou_5cm']:.4f}, "
          f"피크 비율 {bm['peak_ratio']:.3f})")

    print("\n튜닝셋에서 amount 별 (sigma=%.1f)\n" % bs)
    print("| amount | MAE[m] | r | IoU@1cm | IoU@5cm | 피크 비율 |")
    print("|---|---:|---:|---:|---:|---:|")
    for s, a, m in rows:
        if s == bs:
            print(f"| {a} | {m['mae_m']:.4f} | {m['r']:.4f} | {m['iou_1cm']:.4f} "
                  f"| {m['iou_5cm']:.4f} | {m['peak_ratio']:.3f} |")

    test = collect(pred, gt_src, contours_dir, rain, tval, va, args.test_samples, args.min_depth)
    base = score(test, 1.0, 0.0)
    tuned = score(test, bs, ba)
    print(f"\n## val 결과 (표본 {len(test)}개, 위 파라미터를 그대로 적용)\n")
    print("| 지표 | 후처리 없음 | 언샤프 적용 | 변화 |")
    print("|---|---:|---:|---:|")
    for k, label in (("mae_m", "MAE[m]"), ("r", "상관계수 r"), ("iou_1cm", "IoU@1cm"),
                     ("iou_5cm", "IoU@5cm"), ("peak_ratio", "최대수심 비율")):
        d = tuned[k] - base[k]
        print(f"| {label} | {base[k]:.4f} | {tuned[k]:.4f} | {d:+.4f} |")


if __name__ == "__main__":
    main()
