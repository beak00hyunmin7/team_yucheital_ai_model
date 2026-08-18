"""
서울시 실제 등고선 기반 지형 패치를 여러 곳에서 뽑아 (지형, 강우조건, 최종 수심맵)
학습 데이터셋을 만든다. terrainFlowSim/dataset과 같은 포맷(.npy + meta.json)이라
같은 U-Net 파이프라인에 그대로 합쳐서 쓸 수 있다.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "terrainFlowSim"))
from overland_flow_sim import simulate_overland_flow, mass_balance_check  # noqa: E402

from extract_patch import extract_patch, get_bbox  # noqa: E402


def pick_patch_centers(n, size=96, dx=1.0, min_relief=2.0, seed=0, max_tries=5000):
    xmin, xmax, ymin, ymax = get_bbox()
    margin = size * dx / 2.0 + 30.0
    rng = np.random.default_rng(seed)

    picked = []
    tries = 0
    while len(picked) < n and tries < max_tries:
        tries += 1
        cx = rng.uniform(xmin + margin, xmax - margin)
        cy = rng.uniform(ymin + margin, ymax - margin)
        z = extract_patch(cx, cy, size=size, dx=dx)
        if z is None:
            continue
        relief = float(z.max() - z.min())
        if relief < min_relief:
            continue
        picked.append((cx, cy, z))
        if len(picked) % 10 == 0:
            print(f"  picked {len(picked)}/{n} (tries={tries})")

    print(f"done picking: {len(picked)} patches from {tries} tries")
    return picked


def generate_one(z, seed, dx=1.0):
    rng = np.random.default_rng(seed + 100000)
    rain_rate_mmhr = rng.uniform(10.0, 80.0)
    rain_rate = rain_rate_mmhr / 1000.0 / 3600.0
    rain_duration = rng.uniform(900.0, 3600.0)
    total_time = rain_duration + rng.uniform(1800.0, 5400.0)

    h = simulate_overland_flow(
        z, dx=dx, dt=1.0,
        rain_rate=rain_rate, rain_duration=rain_duration,
        total_time=total_time, k_rate=0.4,
    )

    meta = {
        "seed": seed,
        "dx": dx,
        "rain_rate_mmhr": rain_rate_mmhr,
        "rain_duration_s": rain_duration,
        "total_time_s": total_time,
    }
    meta.update(mass_balance_check(z, h, dx, rain_rate, rain_duration))
    return h, meta


def build_dataset(n_samples=100, out_dir="dataset", size=96, dx=1.0, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    patches = pick_patch_centers(n_samples, size=size, dx=dx, seed=seed)

    index = []
    for i, (cx, cy, z) in enumerate(patches):
        h, meta = generate_one(z, seed=seed + i, dx=dx)
        meta["center_x"] = cx
        meta["center_y"] = cy
        meta["relief_m"] = float(z.max() - z.min())
        meta["source"] = "seoul_contour_5000"

        sample_dir = os.path.join(out_dir, f"sample_{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)
        np.save(os.path.join(sample_dir, "terrain.npy"), z.astype(np.float32))
        np.save(os.path.join(sample_dir, "depth.npy"), h.astype(np.float32))
        with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        index.append({"sample": f"sample_{i:04d}", **meta})
        print(f"[{i+1}/{n_samples}] center=({cx:.0f},{cy:.0f}) relief={meta['relief_m']:.1f}m "
              f"rain={meta['rain_rate_mmhr']:.1f}mm/hr max_depth={h.max():.4f}m")

    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\n완료: {len(patches)}개 샘플 -> {out_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--out", type=str, default="dataset")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    build_dataset(n_samples=args.n, out_dir=args.out, size=args.size, seed=args.seed)
