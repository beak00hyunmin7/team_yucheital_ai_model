"""
서울/부산/강원 OpenFOAM 데이터셋(terrain.npy + depth.npy + meta.json)을
ai_model이 기대하는 (등고선, 배수) PNG 이미지 쌍으로 변환하고, 지형은 방향
대칭(배수는 회전/반전해도 물리적으로 동일)이라는 점을 이용해 8배(D4군: 회전
90/180/270 + 각각 좌우반전) 증강한다.

사용법 (ai_model/ 디렉터리에서):
    python scripts/prepare_openfoam_dataset.py
"""

import os
import sys
import json
import glob

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.render_fields import render_contour_rgb, render_depth_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(HERE, "..", "..", "..")  # .../유체이탈/project

REGIONS = {
    "seoul": os.path.join(PROJECT_ROOT, "seoulOpenFOAMDataset", "dataset"),
    "busan": os.path.join(PROJECT_ROOT, "busanOpenFOAMDataset", "dataset"),
    "gangwon": os.path.join(PROJECT_ROOT, "gangwonOpenFOAMDataset", "dataset"),
}

CONTOURS_DIR = os.path.join(HERE, "..", "data", "contours")
FLOW_DIR = os.path.join(HERE, "..", "data", "flow")


def d4_transforms():
    """8개 (이름, 변환함수) - 배열(ny,nx)에 적용, D4 대칭군 전체."""
    def rot(k):
        return lambda a: np.rot90(a, k)

    def flip_rot(k):
        return lambda a: np.rot90(np.fliplr(a), k)

    return [
        ("r0", rot(0)), ("r1", rot(1)), ("r2", rot(2)), ("r3", rot(3)),
        ("f0", flip_rot(0)), ("f1", flip_rot(1)), ("f2", flip_rot(2)), ("f3", flip_rot(3)),
    ]


def process_sample(region, sample_dir, sample_name, dx):
    z = np.load(os.path.join(sample_dir, "terrain.npy"))
    depth = np.load(os.path.join(sample_dir, "depth.npy"))
    ny, nx = z.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    X, Y = np.meshgrid(x, y)

    n_saved = 0
    for aug_name, fn in d4_transforms():
        z_t = fn(z)
        depth_t = fn(depth)

        contour_rgb = render_contour_rgb(X, Y, z_t)
        depth_rgb = render_depth_rgb(depth_t)

        out_name = f"{region}_{sample_name}_{aug_name}.png"
        Image.fromarray(contour_rgb).save(os.path.join(CONTOURS_DIR, out_name))
        Image.fromarray(depth_rgb).save(os.path.join(FLOW_DIR, out_name))
        n_saved += 1
    return n_saved


def main():
    os.makedirs(CONTOURS_DIR, exist_ok=True)
    os.makedirs(FLOW_DIR, exist_ok=True)

    total = 0
    for region, dataset_dir in REGIONS.items():
        sample_dirs = sorted(glob.glob(os.path.join(dataset_dir, "sample_*")))
        print(f"[{region}] {len(sample_dirs)}개 샘플 발견 ({dataset_dir})")
        for i, sdir in enumerate(sample_dirs):
            name = os.path.basename(sdir)
            meta_path = os.path.join(sdir, "meta.json")
            dx = 10.416666666666666
            if os.path.exists(meta_path):
                meta = json.load(open(meta_path, encoding="utf-8"))
                dx = meta.get("dx", dx)
            n = process_sample(region, sdir, name, dx)
            total += n
            print(f"  [{i+1}/{len(sample_dirs)}] {name}: {n}장 저장")

    print(f"\n완료: 총 {total}장 이미지 쌍 -> {CONTOURS_DIR}, {FLOW_DIR}")


if __name__ == "__main__":
    main()
