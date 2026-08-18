"""
(언덕 지형, 강우조건) -> 최종 수심맵 쌍을 대량으로 생성해서
AI 서로게이트 모델(U-Net 등) 학습용 데이터셋으로 저장한다.

각 샘플마다:
  - terrain.npy : (size, size) 고도 [m]
  - rain.npy    : (2,) [rain_rate(m/s), rain_duration(s)]  -- 조건도 입력 채널로 쓸 수 있게 별도 저장
  - depth.npy   : (size, size) 최종 수심 [m]  (AI가 예측해야 할 타깃)

samples/sample_XXXX/ 폴더 아래에 저장.
"""

import os
import json
import numpy as np

from generate_terrain import generate_hill_terrain
from overland_flow_sim import simulate_overland_flow, mass_balance_check


def generate_one_sample(seed, size=96, dx=1.0):
    z, X, Y = generate_hill_terrain(size=size, dx=dx, seed=seed)

    rng = np.random.default_rng(seed + 100000)
    rain_rate_mmhr = rng.uniform(10.0, 80.0)          # 10~80 mm/hr
    rain_rate = rain_rate_mmhr / 1000.0 / 3600.0        # m/s
    rain_duration = rng.uniform(900.0, 3600.0)          # 15~60분
    total_time = rain_duration + rng.uniform(1800.0, 5400.0)  # 강우 후 배수시간 30~90분 추가

    h = simulate_overland_flow(
        z, dx=dx, dt=1.0,
        rain_rate=rain_rate, rain_duration=rain_duration,
        total_time=total_time, k_rate=0.4,
    )

    meta = {
        "seed": seed,
        "size": size,
        "dx": dx,
        "rain_rate_mmhr": rain_rate_mmhr,
        "rain_duration_s": rain_duration,
        "total_time_s": total_time,
    }
    meta.update(mass_balance_check(z, h, dx, rain_rate, rain_duration))
    return z, h, meta


def build_dataset(n_samples=20, out_dir="dataset", size=96, dx=1.0, start_seed=0):
    os.makedirs(out_dir, exist_ok=True)
    index = []

    for i in range(n_samples):
        seed = start_seed + i
        z, h, meta = generate_one_sample(seed, size=size, dx=dx)

        sample_dir = os.path.join(out_dir, f"sample_{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)
        np.save(os.path.join(sample_dir, "terrain.npy"), z.astype(np.float32))
        np.save(os.path.join(sample_dir, "depth.npy"), h.astype(np.float32))
        with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        index.append({"sample": f"sample_{i:04d}", **meta})
        print(f"[{i+1}/{n_samples}] seed={seed} rain={meta['rain_rate_mmhr']:.1f}mm/hr "
              f"dur={meta['rain_duration_s']/60:.0f}min max_depth={h.max():.4f}m "
              f"outflow_ratio={meta['outflow_ratio']:.2f}")

    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {n_samples}개 샘플 -> {out_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--out", type=str, default="dataset")
    parser.add_argument("--start_seed", type=int, default=0)
    args = parser.parse_args()

    build_dataset(n_samples=args.n, out_dir=args.out, size=args.size, start_seed=args.start_seed)
