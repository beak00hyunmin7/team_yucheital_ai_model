"""
파이프라인 자체의 정보 손실을 측정한다 (IMPROVEMENT_GUIDE.md Phase 0 부록).

evaluate.py에서 "최대 수심을 절반으로 예측한다(비율 0.49)"가 나왔는데, 이 손실이
모델 탓인지 표현 방식(수심 -> sqrt 정규화 -> Blues 8bit RGB -> 역LUT -> 수심) 탓인지
구분해야 한다. 구분 방법은 간단하다: **정답을 모델 없이 같은 파이프라인에 통과시킨다.**

    depth(m) --depth_to_norm--> [0,1] --render_depth_rgb--> RGB --역LUT--> norm --norm_to_depth--> depth'(m)

여기서 depth' 와 depth 의 차이 = 모델이 완벽해도 절대 못 넘는 천장.
이 천장이 이미 크다면 아키텍처/데이터가 아니라 표현 방식을 고쳐야 한다.

단계별로 끊어서 어디서 날아가는지도 같이 본다:
  (a) sqrt 정규화 + 역변환만           - 수치 왕복
  (b) + 256px RGB 렌더 + 역LUT         - 렌더링/양자화
  (c) + 96px로 되돌리는 리사이즈        - 배포 경로 전체 (evaluate.py가 재는 것)

사용법:
    .venv\\Scripts\\python.exe scripts\\diagnose_pipeline.py --samples 60
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.utils.render_fields import (depth_to_norm, norm_to_depth, render_depth_rgb,
                                     DEPTH_VMAX_M)  # noqa: E402
from prepare_openfoam_dataset import REGIONS, _pick_timesteps  # noqa: E402

_LUT_N = 256
import matplotlib  # noqa: E402
matplotlib.use("Agg")
_LUT = (np.array([matplotlib.colormaps["Blues"](i / (_LUT_N - 1))[:3]
                  for i in range(_LUT_N)]) * 255.0).astype(np.float32)
_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_LUT_T = torch.from_numpy(_LUT).to(_DEV)


def lut_decode(rgb_uint8):
    """Blues RGB -> [0,1] 정규화값 (inference_api._image_to_depth와 동일한 최근접 탐색)."""
    arr = torch.from_numpy(np.asarray(rgb_uint8).astype(np.float32)).to(_DEV)
    hw = arr.shape[:2]
    d2 = ((arr.reshape(-1, 1, 3) - _LUT_T.reshape(1, -1, 3)) ** 2).sum(-1)
    return (d2.argmin(1).reshape(hw).float() / (_LUT_N - 1)).cpu().numpy()


def stats(name, ref, got):
    """ref 대비 got 의 손실 요약. peak/mean 비율이 핵심 지표.

    단계 (b)는 렌더 해상도(256)라 원본(96)과 격자가 다르다. 비율 지표는 격자와
    무관하게 의미가 있으므로 그대로 내고, 픽셀 대응이 필요한 MAE/재현율만 생략한다.
    """
    same = ref.shape == got.shape
    return {
        "stage": name,
        "mae_m": float(np.abs(got - ref).mean()) if same else float("nan"),
        "peak_ratio": float(got.max() / max(ref.max(), 1e-9)),
        "mean_ratio": float(got.mean() / max(ref.mean(), 1e-9)),
        # 전체가 1cm 미만으로 마른 샘플은 재현율이 정의되지 않는다 -> 평균에서 제외
        "wet1cm_recall": (float(((got >= 0.01) & (ref >= 0.01)).sum() / (ref >= 0.01).sum())
                          if same and (ref >= 0.01).any() else float("nan")),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=60)
    p.add_argument("--grid", type=int, default=96, help="원본 격자 크기(되돌릴 목표 해상도)")
    args = p.parse_args()

    # 지역별로 고르게 뽑는다 (gangwon 이 적어서 무작위로 뽑으면 거의 안 걸린다)
    picked = []
    per_region = max(1, args.samples // len(REGIONS))
    for region, ddir in REGIONS.items():
        dirs = sorted(glob.glob(os.path.join(ddir, "sample_*")))
        step = max(1, len(dirs) // per_region)
        picked += [(region, d) for d in dirs[::step][:per_region]]

    acc = {}
    n = 0
    for region, sdir in picked:
        meta_path = os.path.join(sdir, "meta.json")
        meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
        steps = _pick_timesteps(sdir, meta.get("total_time_s", 120.0))
        depth = np.clip(steps[-1][0], 0.0, DEPTH_VMAX_M)   # 마지막 시점(물이 가장 많이 모인 상태)
        if depth.max() < 1e-4:
            continue

        norm = depth_to_norm(depth)

        # (a) 수치 왕복만
        a = norm_to_depth(norm)

        # (b) + RGB 렌더 + 역LUT (렌더 해상도 256)
        rgb = render_depth_rgb(norm, vmax=1.0)
        b = norm_to_depth(lut_decode(rgb))

        # (c) + 원본 격자로 리사이즈 (배포 경로 전체)
        img = Image.fromarray(rgb).resize((args.grid, args.grid), Image.BICUBIC)
        c = norm_to_depth(lut_decode(img))
        c = np.flipud(c)  # render_depth_rgb 는 origin="lower" 라 이미지가 상하반전돼 있다

        for st in (stats("(a) sqrt 정규화 왕복", depth, a),
                   stats("(b) + RGB 렌더 + 역LUT", depth, b),
                   stats("(c) + 원본격자 리사이즈 = 배포경로", depth, c)):
            acc.setdefault(st["stage"], []).append(st)
        n += 1

    print(f"표본 {n}개 (지역별 균등 추출, 마지막 시점)\n")
    print("| 단계 | MAE[m] | 최대수심 비율 | 평균수심 비율 | 젖은영역(1cm) 재현율 |")
    print("|---|---:|---:|---:|---:|")
    for stage, items in acc.items():
        g = lambda k: np.nanmean([x[k] for x in items])  # noqa: E731
        fmt = lambda v, d=4: "—" if np.isnan(v) else f"{v:.{d}f}"  # noqa: E731
        print(f"| {stage} | {fmt(g('mae_m'))} | {fmt(g('peak_ratio'), 3)} | "
              f"{fmt(g('mean_ratio'), 3)} | {fmt(g('wet1cm_recall'), 3)} |")
    print("\n해석: 비율 1.000 에 가까울수록 손실 없음. (c)의 최대수심 비율이 모델이 "
          "완벽할 때 도달 가능한 상한이다.")


if __name__ == "__main__":
    main()
