"""
피크 과소예측의 성격을 가른다: "번짐(blur)" 인가 "축소(magnitude)" 인가.

evaluate.py 가 최대수심 비율 0.62(천장 0.93)를 보고했는데, 같은 숫자가 두 가지 서로
다른 실패에서 나올 수 있다.

  H1 번짐  — 물의 총량은 맞는데 넓게 퍼뜨려서 봉우리만 낮다.
             처방: 해상도/선명도 (더 높은 해상도, 주파수 손실, 스킵 연결 강화)
  H2 축소  — 깊은 영역을 전반적으로 작게 예측한다. 퍼짐과 무관.
             처방: 손실 가중/출력 표현 (이미 wet_weight 로 절반 회수, 포화)

가르는 방법: **정답을 인위적으로 흐리게 만들어 예측과 맞춰본다.**
어떤 흐림 정도 sigma 에서 blur(정답, sigma) 가 예측과 가장 가까워지는지 찾고,
그때 blur(정답) 의 최대수심 비율이 예측의 0.62 를 설명하는지 본다.

  - 설명된다  -> 모델은 사실상 "흐려진 정답"을 내고 있다 = H1
  - 설명 안 됨 -> 흐림만으로는 부족하고 크기 자체가 작다 = H2

보조로 정답 수심 구간별(십분위) 예측/정답 비를 같이 낸다. H2 라면 깊은 구간에서
비율이 계통적으로 낮게 나온다.

사용법:
    .venv\\Scripts\\python.exe scripts\\diagnose_sharpness.py --checkpoint checkpoints_p2_wet8/best.pt \
        --config configs/config_p2_wet8.yaml --samples 60
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

from src.datasets.contour_flow_dataset import split_train_val, _group_key  # noqa: E402
from src.ground_truth import GroundTruth, to_array_orientation  # noqa: E402
from evaluate import Predictor, NAME_RE  # noqa: E402

SIGMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_p2_wet8/best.pt")
    ap.add_argument("--config", default="configs/config_p2_wet8.yaml")
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--min-depth", type=float, default=0.05,
                    help="정답 최대수심이 이보다 얕은 케이스는 제외 (비교 정보가 없음)")
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    contours_dir = os.path.join(ROOT, cfg["contours_dir"])
    flow_dir = os.path.join(ROOT, cfg["flow_dir"])

    _, val_names = split_train_val(contours_dir, flow_dir,
                                   val_split=cfg["val_split"], seed=cfg["seed"])
    names = sorted(n for n in val_names if "_r0_t" in n)

    rain = json.load(open(os.path.join(ROOT, cfg["rain_values_json"]), encoding="utf-8"))
    tval = json.load(open(os.path.join(ROOT, cfg["time_values_json"]), encoding="utf-8"))

    pred = Predictor(os.path.join(ROOT, args.checkpoint), cfg["image_size"])
    gt_src = GroundTruth()
    print(f"체크포인트 {args.checkpoint} (image_size={pred.image_size}, cond_dim={pred.cond_dim})")

    # sigma 별 누적값
    best_sigmas, rows = [], []
    blur_peak = {s: [] for s in SIGMAS}
    pred_peak = []
    decile_p = np.zeros(10)
    decile_g = np.zeros(10)
    decile_n = np.zeros(10)
    used = 0

    for n in names:
        if used >= args.samples:
            break
        g = gt_src.get(n)
        if g is None:
            continue
        gt = g["depth_m"]
        if gt.max() < args.min_depth:
            continue

        p = to_array_orientation(
            pred.to_depth(pred.forward([os.path.join(contours_dir, n)],
                                       [pred.cond(rain.get(n), tval.get(n))]), gt.shape)[0])

        # 1) 어떤 sigma 로 정답을 흐리면 예측과 가장 가까워지는가
        errs = []
        for s in SIGMAS:
            b = gaussian_filter(gt, s) if s > 0 else gt
            errs.append(float(np.abs(b - p).mean()))
            blur_peak[s].append(float(b.max() / max(gt.max(), 1e-9)))
        best_sigmas.append(SIGMAS[int(np.argmin(errs))])
        pred_peak.append(float(p.max() / max(gt.max(), 1e-9)))

        # 2) 정답 수심 십분위별 예측/정답 평균
        wet = gt[gt >= 0.001]
        if wet.size > 50:
            edges = np.quantile(wet, np.linspace(0, 1, 11))
            for d in range(10):
                m = (gt >= edges[d]) & (gt < edges[d + 1] if d < 9 else gt <= edges[10])
                if m.sum():
                    decile_g[d] += gt[m].mean() * m.sum()
                    decile_p[d] += p[m].mean() * m.sum()
                    decile_n[d] += m.sum()

        rows.append((n, float(gt.max()), float(p.max())))
        used += 1

    if not used:
        print("평가할 샘플이 없습니다.")
        return

    bs = np.array(best_sigmas)
    pp = float(np.mean(pred_peak))
    print(f"\n표본 {used}개 (정답 최대수심 {args.min_depth}m 이상인 val 케이스)\n")

    print("## 1. 정답을 흐리게 만들어 예측과 맞추기\n")
    print("| 흐림 sigma (96격자 픽셀) | blur(정답)의 최대수심 비율 |")
    print("|---|---:|")
    for s in SIGMAS:
        print(f"| {s:.2f} | {np.mean(blur_peak[s]):.3f} |")
    print(f"\n예측의 최대수심 비율: **{pp:.3f}**")
    print(f"예측과 가장 가까워지는 sigma: 중앙값 {np.median(bs):.2f}, 평균 {bs.mean():.2f}")

    # 예측 peak 를 설명하는 sigma 를 보간으로 추정
    xs = np.array(SIGMAS)
    ys = np.array([np.mean(blur_peak[s]) for s in SIGMAS])
    if ys.min() <= pp <= ys.max():
        sigma_peak = float(np.interp(pp, ys[::-1], xs[::-1]))
        print(f"예측의 최대수심 비율({pp:.3f})을 내려면 정답을 sigma≈{sigma_peak:.2f} 로 흐려야 함")
    else:
        sigma_peak = None
        print(f"경고: 예측 비율 {pp:.3f} 이 흐림만으로 설명되는 범위 밖")

    print("\n## 2. 정답 수심 십분위별 예측/정답 비\n")
    print("| 십분위 | 정답 평균 [m] | 예측 평균 [m] | 비율 |")
    print("|---|---:|---:|---:|")
    for d in range(10):
        if decile_n[d]:
            gm, pm = decile_g[d] / decile_n[d], decile_p[d] / decile_n[d]
            print(f"| {d+1} ({'얕음' if d < 3 else '깊음' if d > 6 else '중간'}) "
                  f"| {gm:.4f} | {pm:.4f} | {pm/max(gm,1e-9):.3f} |")

    print("\n## 판정\n")
    med = float(np.median(bs))
    if sigma_peak is not None and abs(sigma_peak - med) < 0.8:
        print(f"흐림 정도({med:.2f})와 피크를 설명하는 흐림({sigma_peak:.2f})이 일치한다.")
        print("-> **H1 번짐**. 모델은 사실상 '흐려진 정답'을 내고 있다.")
        print("   처방은 선명도 쪽: 해상도 정합, 고주파 손실, 스킵 연결 강화.")
    else:
        print(f"최적 흐림({med:.2f})만으로는 피크 비율이 설명되지 않는다"
              + (f" (필요 sigma {sigma_peak:.2f})" if sigma_peak else "") + ".")
        print("-> **H2 축소** 성분이 크다. 깊은 구간의 크기 자체를 못 낸다.")
        print("   십분위 표에서 깊은 쪽 비율이 낮으면 확정.")


if __name__ == "__main__":
    main()
