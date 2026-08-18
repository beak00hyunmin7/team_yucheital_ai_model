"""
부산 지역 학습 데이터셋용 후보 지형 N개를 미리 골라서(완만한 곳만 필터링) 각 샘플
폴더에 terrain_raw.npy + params.json(강우 조건, 권장 시뮬레이션 시간)을 저장해둔다.
seoulOpenFOAMDataset/prepare_samples.py와 동일 로직, busanTerrain 포인트클라우드 사용.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "busanTerrain"))
from extract_patch import extract_patch, get_bbox  # noqa: E402

from estimate_duration import estimate_total_time  # noqa: E402

SIZE = 96
DOMAIN_M = 1000.0
DX = DOMAIN_M / SIZE  # ~10.4167 m

STAGING_DIR = os.path.join(os.path.dirname(__file__), "staging")


def pick_and_prepare(n_samples, max_slope=0.35, min_relief=2.0, seed=0, max_tries=4000):
    xmin, xmax, ymin, ymax = get_bbox()
    margin = DOMAIN_M / 2.0 + 30.0
    rng = np.random.default_rng(seed)

    os.makedirs(STAGING_DIR, exist_ok=True)
    accepted = []
    tries = 0
    while len(accepted) < n_samples and tries < max_tries:
        tries += 1
        cx = rng.uniform(xmin + margin, xmax - margin)
        cy = rng.uniform(ymin + margin, ymax - margin)
        z = extract_patch(cx, cy, size=SIZE, dx=DX)
        if z is None:
            continue
        if float(z.max() - z.min()) < min_relief:
            continue  # 바다/매립지 등 완전 평지 배제
        gy, gx = np.gradient(z, DX)
        maxslope = float(np.sqrt(gx ** 2 + gy ** 2).max())
        if maxslope > max_slope:
            continue

        total_time, mean_slope, L = estimate_total_time(z, DX)
        rain_mm = float(rng.uniform(20.0, 80.0))
        film_depth = rain_mm / 1000.0

        i = len(accepted)
        sample_dir = os.path.join(STAGING_DIR, f"sample_{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)
        np.save(os.path.join(sample_dir, "terrain_raw.npy"), z.astype(np.float32))

        params = {
            "sample": f"sample_{i:04d}",
            "center_x": cx, "center_y": cy,
            "dx": DX, "size": SIZE,
            "relief_m": float(z.max() - z.min()),
            "max_slope": maxslope,
            "mean_slope": mean_slope,
            "total_time_s": total_time,
            "rain_mm": rain_mm,
            "film_depth_m": film_depth,
            "source": "busan_national_contour",
        }
        with open(os.path.join(sample_dir, "params.json"), "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

        accepted.append(params)
        print(f"[{i+1}/{n_samples}] center=({cx:.0f},{cy:.0f}) relief={params['relief_m']:.1f}m "
              f"maxslope={maxslope:.3f} total_time={total_time:.0f}s rain={rain_mm:.0f}mm "
              f"(tries so far: {tries})")

    with open(os.path.join(STAGING_DIR, "index.json"), "w", encoding="utf-8") as f:
        json.dump(accepted, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {len(accepted)}/{n_samples} 샘플 준비됨 ({tries} 시도) -> {STAGING_DIR}")
    return accepted


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--max-slope", type=float, default=0.35)
    p.add_argument("--min-relief", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    pick_and_prepare(args.n, max_slope=args.max_slope, min_relief=args.min_relief, seed=args.seed)
