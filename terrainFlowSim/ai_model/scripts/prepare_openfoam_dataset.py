"""
서울/부산/강원 OpenFOAM 데이터셋(terrain.npy + depth.npy + meta.json)을
ai_model이 기대하는 (등고선, 배수) PNG 이미지 쌍으로 변환하고, 지형은 방향
대칭(배수는 회전/반전해도 물리적으로 동일)이라는 점을 이용해 8배(D4군: 회전
90/180/270 + 각각 좌우반전) 증강한다.

시점(time) 샘플링: run_batch.py가 전체 타임스텝을 reconstruct해서 저장해둔
depth_series.npy/times.npy가 있는 샘플(주로 강우 변형 rv1/rv2 이후 생성분)은
전체 시뮬레이션 구간에서 대표 시점 여러 개(TIME_FRACTIONS)를 뽑아 각각 별도
이미지 쌍으로 저장한다 - 같은 지형·강우량이라도 시점에 따라 물이 차오르는
패턴이 달라지므로 강우 조건화(FiLM)가 조건을 무시하기 어렵게 만드는 효과와
학습 쌍 수를 늘리는 효과를 같이 노린다. depth_series.npy가 없는 예전 샘플은
기존처럼 depth.npy 하나(= 시뮬레이션 종료 시점)만 사용한다.

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
from src.utils.render_fields import (render_contour_rgb, render_depth_rgb, render_rain_channel,
                                      depth_to_norm)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(HERE, "..", "..", "..")  # .../유체이탈/project

REGIONS = {
    "seoul": os.path.join(PROJECT_ROOT, "seoulOpenFOAMDataset", "dataset"),
    "busan": os.path.join(PROJECT_ROOT, "busanOpenFOAMDataset", "dataset"),
    "gangwon": os.path.join(PROJECT_ROOT, "gangwonOpenFOAMDataset", "dataset"),
}

CONTOURS_DIR = os.path.join(HERE, "..", "data", "contours")
FLOW_DIR = os.path.join(HERE, "..", "data", "flow")
RAIN_DIR = os.path.join(HERE, "..", "data", "rain")
RAIN_VALUES_JSON = os.path.join(HERE, "..", "data", "rain_values.json")
TIME_VALUES_JSON = os.path.join(HERE, "..", "data", "time_values.json")

# 시뮬레이션 총 시간(120s) 대비 뽑을 대표 시점 비율. 초반(물이 거의 없는 구간)
# 비중을 줄이고 중후반에 몰아서, "항상 거의 0을 예측"하는 게으른 답을 배우기
# 어렵게 하면서도 시점 수(=이미지 수 배수)는 6개로 억제했다.
TIME_FRACTIONS = [0.15, 0.30, 0.45, 0.60, 0.80, 1.0]


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


def _pick_timesteps(sample_dir, total_time):
    """(depth 배열, 실제 시각[s]) 목록을 시점 개수만큼 뽑아 반환한다.

    depth_series.npy/times.npy가 있으면 TIME_FRACTIONS 각각에 가장 가까운
    실제 저장 시점을 골라 여러 장 반환하고, 없으면(예전 샘플) depth.npy
    하나(= total_time 시점)만 담은 1개짜리 목록을 반환해 하위호환을 유지한다.
    """
    series_path = os.path.join(sample_dir, "depth_series.npy")
    times_path = os.path.join(sample_dir, "times.npy")
    if os.path.exists(series_path) and os.path.exists(times_path):
        series = np.load(series_path)
        times = np.load(times_path)
        picked = []
        for frac in TIME_FRACTIONS:
            idx = int(np.argmin(np.abs(times - frac * total_time)))
            picked.append((series[idx], float(times[idx])))
        return picked
    depth = np.load(os.path.join(sample_dir, "depth.npy"))
    return [(depth, float(total_time))]


def process_sample(region, sample_dir, sample_name, dx, total_time, rain_mm=None,
                    rain_values=None, time_values=None, flow_only=False):
    """flow_only=True면 등고선/강우 이미지는 건드리지 않고 배수(타깃) 이미지만 다시 만든다.
    타깃 정규화 방식만 바꿔서 재생성할 때 등고선 렌더링(가장 비싼 단계)을 건너뛰기 위함."""
    z = np.load(os.path.join(sample_dir, "terrain.npy"))
    ny, nx = z.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    X, Y = np.meshgrid(x, y)

    rain_img = None
    if rain_mm is not None and not flow_only:
        rain_img = Image.fromarray(render_rain_channel(rain_mm), mode="L")

    timesteps = _pick_timesteps(sample_dir, total_time)

    n_saved = 0
    for t_idx, (depth, t_val) in enumerate(timesteps):
        for aug_name, fn in d4_transforms():
            z_t = fn(z)
            depth_t = fn(depth)

            # 타깃은 전역 기준(DEPTH_VMAX_M) + sqrt 압축으로 정규화한 뒤 렌더링한다.
            # 예전처럼 샘플별 자체 정규화를 하면 절대 수심 정보가 사라져서
            # 강우량 조건화를 아예 학습할 수 없다 (render_fields.py 주석 참고).
            depth_rgb = render_depth_rgb(depth_to_norm(depth_t), vmax=1.0)

            # 시점이 1개뿐인(예전 샘플) 경우도 접미사를 항상 붙여서 파일명 규칙을
            # 통일한다 - time_values.json에 항상 키가 존재해야 시점 조건화
            # 학습(cond_dim=2)에서 필터링이 일관되게 동작한다.
            out_name = f"{region}_{sample_name}_{aug_name}_t{t_idx}.png"
            Image.fromarray(depth_rgb).save(os.path.join(FLOW_DIR, out_name))
            if not flow_only:
                contour_rgb = render_contour_rgb(X, Y, z_t)
                Image.fromarray(contour_rgb).save(os.path.join(CONTOURS_DIR, out_name))
            if rain_img is not None:
                # 강우 채널은 방향성이 없는 값이라 회전/반전해도 동일 -> 그대로 재사용
                rain_img.save(os.path.join(RAIN_DIR, out_name))
            if rain_values is not None and rain_mm is not None:
                # FiLM 방식용: 이미지가 아니라 {파일명: 강우량mm} 매핑으로도 저장
                rain_values[out_name] = rain_mm
            if time_values is not None:
                time_values[out_name] = t_val
            n_saved += 1
    return n_saved


def main(flow_only=False):
    os.makedirs(CONTOURS_DIR, exist_ok=True)
    os.makedirs(FLOW_DIR, exist_ok=True)
    os.makedirs(RAIN_DIR, exist_ok=True)

    total = 0
    n_with_series = 0
    rain_values = {}
    time_values = {}
    for region, dataset_dir in REGIONS.items():
        sample_dirs = sorted(glob.glob(os.path.join(dataset_dir, "sample_*")))
        print(f"[{region}] {len(sample_dirs)}개 샘플 발견 ({dataset_dir})")
        for i, sdir in enumerate(sample_dirs):
            name = os.path.basename(sdir)
            meta_path = os.path.join(sdir, "meta.json")
            dx = 10.416666666666666
            rain_mm = None
            total_time = 120.0
            if os.path.exists(meta_path):
                meta = json.load(open(meta_path, encoding="utf-8"))
                dx = meta.get("dx", dx)
                rain_mm = meta.get("rain_mm")
                total_time = meta.get("total_time_s", total_time)
            if os.path.exists(os.path.join(sdir, "depth_series.npy")):
                n_with_series += 1
            n = process_sample(region, sdir, name, dx, total_time, rain_mm=rain_mm,
                                rain_values=rain_values, time_values=time_values,
                                flow_only=flow_only)
            total += n
            print(f"  [{i+1}/{len(sample_dirs)}] {name}: {n}장 저장")

    with open(RAIN_VALUES_JSON, "w", encoding="utf-8") as f:
        json.dump(rain_values, f, ensure_ascii=False, indent=2)
    with open(TIME_VALUES_JSON, "w", encoding="utf-8") as f:
        json.dump(time_values, f, ensure_ascii=False, indent=2)

    print(f"\n완료: 총 {total}장 이미지 쌍 -> {CONTOURS_DIR}, {FLOW_DIR}, {RAIN_DIR}")
    print(f"강우량 매핑({len(rain_values)}개) -> {RAIN_VALUES_JSON}")
    print(f"시점 매핑({len(time_values)}개, 다중시점 샘플 {n_with_series}개) -> {TIME_VALUES_JSON}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--flow-only", action="store_true",
                    help="등고선/강우 이미지는 그대로 두고 배수(타깃) 이미지만 다시 생성")
    args = p.parse_args()
    main(flow_only=args.flow_only)
