"""
타입 조합 지형 생성기(terrain_types.py) + 빠른 지표유출 근사 시뮬레이터
(terrainFlowSim/overland_flow_sim.py, hillTerrainCase_rain에서 쓴 것과 같은 물리 모델)로
(등고선, 배수) 이미지 쌍을 대량 생성해 ai_model/data/에 저장한다.

실제 OpenFOAM(interFoam)만큼 정확하지는 않지만 샘플당 몇 초면 만들어지므로
학습 파이프라인을 먼저 검증하고 데이터 양을 빠르게 늘리는 용도로 쓴다.

사용법 (ai_model/ 디렉터리에서):
    python scripts/generate_dataset.py --n 200 --size 96
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.terrain_types import generate_hill_terrain_v2
from src.utils.render_fields import render_contour_rgb, render_depth_rgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "terrainFlowSim"))
from overland_flow_sim import simulate_overland_flow  # noqa: E402


def generate_one(seed, size, dx):
    z, X, Y = generate_hill_terrain_v2(size=size, dx=dx, seed=seed)

    rng = np.random.default_rng(seed + 100000)
    rain_rate_mmhr = rng.uniform(10.0, 80.0)
    rain_rate = rain_rate_mmhr / 1000.0 / 3600.0
    rain_duration = rng.uniform(900.0, 3600.0)
    total_time = rain_duration + rng.uniform(1800.0, 5400.0)

    depth = simulate_overland_flow(
        z, dx=dx, dt=1.0,
        rain_rate=rain_rate, rain_duration=rain_duration,
        total_time=total_time, k_rate=0.4,
    )
    return X, Y, z, depth


def main(args):
    os.makedirs(args.contours_dir, exist_ok=True)
    os.makedirs(args.flow_dir, exist_ok=True)

    for i in range(args.n):
        seed = args.start_seed + i
        X, Y, z, depth = generate_one(seed, args.size, args.dx)

        contour_rgb = render_contour_rgb(X, Y, z)
        depth_rgb = render_depth_rgb(depth)

        name = f"{args.name_prefix}_{seed:04d}"
        Image.fromarray(contour_rgb).save(os.path.join(args.contours_dir, f"{name}.png"))
        Image.fromarray(depth_rgb).save(os.path.join(args.flow_dir, f"{name}.png"))

        print(f"[{i + 1}/{args.n}] seed={seed} max_depth={depth.max():.4f}m saved -> {name}.png")

    print(f"\n완료: {args.n}개 샘플 -> {args.contours_dir}/, {args.flow_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--name_prefix", type=str, default="terrain")
    parser.add_argument("--contours_dir", type=str, default="data/contours")
    parser.add_argument("--flow_dir", type=str, default="data/flow")
    args = parser.parse_args()

    main(args)
