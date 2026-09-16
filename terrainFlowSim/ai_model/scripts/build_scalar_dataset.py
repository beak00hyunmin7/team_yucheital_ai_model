"""
B안: 스칼라 격자 회귀용 데이터셋 빌더.

OpenFOAM 원본 샘플( {region}OpenFOAMDataset/dataset/sample_*/{terrain,depth}.npy + meta.json )을
이미지로 렌더링하지 않고 그대로 모아서 data_scalar/ 에 저장한다.

  data_scalar/
    samples/<region>_<sample>[_rvN].npz   # terrain(H,W f32), depth(H,W f32), rain_mm
    stats.json                            # 정규화 상수 + train/val 분할(지형 기준, 누수 방지)

이미지 파이프라인 대비 장점:
  - matplotlib 렌더/역해석에서 잃던 정밀도를 그대로 보존
  - 입력/출력 1채널 -> 모델이 더 작아짐, "terrain 컬러맵 디코딩"을 안 배워도 됨
  - depth 가 sub-cm 이라 분포가 극단적으로 치우침 -> log 정규화로 학습 가능하게 만듦

사용법 (ai_model/ 디렉터리에서):
    python scripts/build_scalar_dataset.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AI_MODEL = os.path.join(HERE, "..")
PROJECT_ROOT = os.path.join(AI_MODEL, "..", "..")  # .../유체이탈/project
sys.path.insert(0, os.path.abspath(AI_MODEL))       # src.* import 가능하게

REGIONS = {
    "seoul": os.path.join(PROJECT_ROOT, "seoulOpenFOAMDataset", "dataset"),
    "busan": os.path.join(PROJECT_ROOT, "busanOpenFOAMDataset", "dataset"),
    "gangwon": os.path.join(PROJECT_ROOT, "gangwonOpenFOAMDataset", "dataset"),
}

OUT_DIR = os.path.join(AI_MODEL, "data_scalar")
SAMPLES_DIR = os.path.join(OUT_DIR, "samples")

DEPTH_REF_M = 1e-4          # 기본값. --depth-ref-mm 로 덮어씀 (log 정규화 기준 최소 수심)
DEPTH_PCTL = 99.5          # 이 분위수를 depth 정규화 상한으로 (이상치 컷)
RAIN_MIN_MM, RAIN_MAX_MM = 20.0, 80.0
VAL_SPLIT = 0.15
SEED = 42

_RV_RE = re.compile(r"^(.*)_rv\d+$")


def group_key(region: str, sample_name: str) -> str:
    """rv(강우변형) 접미사를 떼어 같은 지형끼리 묶는다 (train/val 누수 방지)."""
    base = _RV_RE.match(sample_name)
    return f"{region}/{base.group(1) if base else sample_name}"


def add_flowacc() -> None:
    """이미 만든 npz 각각에 D8 흐름누적 채널(flowacc)을 추가 저장한다 (B4 실험용, 1회)."""
    from src.utils.flow_accum import d8_flow_accumulation

    npzs = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.npz")))
    for i, path in enumerate(npzs):
        d = dict(np.load(path))
        if "flowacc" in d:
            continue
        d["flowacc"] = d8_flow_accumulation(d["terrain"]).astype(np.float32)
        np.savez_compressed(path, **d)
        if (i + 1) % 100 == 0:
            print(f"  flowacc {i + 1}/{len(npzs)}")
    print(f"flowacc 채널 추가 완료: {len(npzs)}개")


def main(depth_ref_m: float = DEPTH_REF_M, pctl: float = DEPTH_PCTL, stats_only: bool = False) -> None:
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    print(f"depth_ref = {depth_ref_m * 1000:.3g} mm,  pctl = {pctl},  stats_only = {stats_only}")

    records = []
    all_depth_hi = []     # 정규화 상수 추정용 (nonzero depth 표본)
    terrain = None

    if stats_only:
        # 이미 만들어둔 npz 에서 depth 만 다시 읽어 정규화 상수/분할만 갱신 (빠름)
        for npz in sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.npz"))):
            out_name = os.path.splitext(os.path.basename(npz))[0]
            d = np.load(npz)
            depth = d["depth"].astype(np.float32)
            terrain = d["terrain"]
            region = out_name.split("_", 1)[0]
            sample_name = out_name.split("_", 1)[1]
            records.append({"name": out_name, "group": group_key(region, sample_name),
                            "rain_mm": float(d["rain_mm"])})
            nz = depth[depth > depth_ref_m]
            if nz.size:
                all_depth_hi.append(nz[:: max(1, nz.size // 2000)])
    else:
        for region, ds_dir in REGIONS.items():
            sample_dirs = sorted(glob.glob(os.path.join(ds_dir, "sample_*")))
            print(f"[{region}] {len(sample_dirs)}개 샘플  ({ds_dir})")
            for sdir in sample_dirs:
                name = os.path.basename(sdir)
                tpath, dpath = os.path.join(sdir, "terrain.npy"), os.path.join(sdir, "depth.npy")
                if not (os.path.exists(tpath) and os.path.exists(dpath)):
                    continue
                terrain = np.load(tpath).astype(np.float32)
                depth = np.load(dpath).astype(np.float32)
                if terrain.shape != depth.shape or terrain.ndim != 2:
                    print(f"  ! {name}: shape 불일치 {terrain.shape} vs {depth.shape} - 건너뜀")
                    continue

                rain_mm = None
                mpath = os.path.join(sdir, "meta.json")
                if os.path.exists(mpath):
                    rain_mm = json.load(open(mpath, encoding="utf-8")).get("rain_mm")
                if rain_mm is None:
                    print(f"  ! {name}: rain_mm 없음 - 건너뜀")
                    continue

                out_name = f"{region}_{name}"
                np.savez_compressed(
                    os.path.join(SAMPLES_DIR, out_name + ".npz"),
                    terrain=terrain, depth=np.maximum(depth, 0.0),
                    rain_mm=np.float32(rain_mm),
                )
                records.append({"name": out_name, "group": group_key(region, name),
                                "rain_mm": float(rain_mm)})
                nz = depth[depth > depth_ref_m]
                if nz.size:
                    all_depth_hi.append(nz[:: max(1, nz.size // 2000)])

    if not records:
        raise SystemExit("샘플을 하나도 찾지 못했습니다. REGIONS 경로 또는 data_scalar/samples 를 확인하세요.")

    depth_pool = np.concatenate(all_depth_hi) if all_depth_hi else np.array([1e-3])
    depth_max = float(np.percentile(depth_pool, pctl))
    log_max = float(np.log1p(depth_max / depth_ref_m))

    # flowacc 채널이 이미 있으면 정규화 상수도 계산해 stats 에 넣는다
    flowacc_log_max = None
    fa_pool = []
    for npz in glob.glob(os.path.join(SAMPLES_DIR, "*.npz")):
        z = np.load(npz)
        if "flowacc" in z:
            fa = np.log1p(z["flowacc"].ravel())
            fa_pool.append(fa[:: max(1, fa.size // 500)])
    if fa_pool:
        flowacc_log_max = float(np.percentile(np.concatenate(fa_pool), 99.9))

    # 지형 그룹 기준 train/val 분할
    groups = sorted({r["group"] for r in records})
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(groups))
    n_val = max(1, int(len(groups) * VAL_SPLIT))
    val_groups = {groups[i] for i in perm[:n_val]}
    for r in records:
        r["split"] = "val" if r["group"] in val_groups else "train"

    stats = {
        "count": len(records),
        "n_groups": len(groups),
        "grid_shape": list(terrain.shape),
        "depth_ref_m": depth_ref_m,
        "depth_max_m": depth_max,
        "depth_log_max": log_max,
        "flowacc_log_max": flowacc_log_max,
        "rain_min_mm": RAIN_MIN_MM,
        "rain_max_mm": RAIN_MAX_MM,
        "train": [r["name"] for r in records if r["split"] == "train"],
        "val": [r["name"] for r in records if r["split"] == "val"],
        "records": records,
    }
    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(
        f"\n완료: {len(records)}개 npz -> {SAMPLES_DIR}\n"
        f"  train {len(stats['train'])} / val {len(stats['val'])}  (지형그룹 {len(groups)}개)\n"
        f"  depth 정규화: ref={depth_ref_m} m, max(p{pctl})={depth_max:.5f} m, log_max={log_max:.3f}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-ref-mm", type=float, default=DEPTH_REF_M * 1000,
                    help="log 정규화 기준 최소 수심 (mm). 크게 잡을수록 얇은 필름은 무시하고 채널만 강조")
    ap.add_argument("--pctl", type=float, default=DEPTH_PCTL, help="depth 정규화 상한 분위수")
    ap.add_argument("--stats-only", action="store_true",
                    help="npz 재생성 없이 stats.json(정규화상수+분할)만 갱신")
    ap.add_argument("--add-flowacc", action="store_true",
                    help="기존 npz 각각에 D8 흐름누적 채널 추가 (B4 실험용, 1회)")
    a = ap.parse_args()
    if a.add_flowacc:
        sys.exit(add_flowacc())
    sys.exit(main(depth_ref_m=a.depth_ref_mm / 1000.0, pctl=a.pctl, stats_only=a.stats_only))
