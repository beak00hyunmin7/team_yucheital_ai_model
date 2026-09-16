#!/usr/bin/env python3
"""
이미 뽑아둔 지형(staging/sample_XXXX/terrain_raw.npy)에 강우 조건만 다르게 바꿔서
추가 OpenFOAM 시뮬레이션을 돌린다. 같은 지형 + 다른 강우 -> 다른 결과 쌍을 만들어서
나중에 모델 입력에 강우 조건을 추가할 수 있게 데이터를 보강하는 용도.

run_batch.py의 sh()/read_alpha_field() 등을 재사용한다. WSL 안에서 실행.

사용법:
    python3 run_rain_variants.py --n_source 20 --n_variants 2 --seed 100
"""

import os
import re
import json
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_batch  # sh(), read_alpha_field(), TEMPLATE, WORK, NZ, BOX_H 재사용

BASE = os.path.dirname(os.path.abspath(__file__))
STAGING = os.path.join(BASE, "staging")
DATASET = os.path.join(BASE, "dataset")


def run_one_variant(params, source_name, idx, n_total):
    name = params["sample"]
    case_dir = os.path.join(run_batch.WORK, name)
    log = os.path.join(BASE, "batch_log_rain_variants.txt")

    def L(msg):
        line = f"[{idx}/{n_total}] {name}: {msg}"
        print(line, flush=True)
        with open(log, "a") as f:
            f.write(line + "\n")

    t0 = time.time()
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    shutil.copytree(run_batch.TEMPLATE, case_dir)
    shutil.copy(os.path.join(STAGING, source_name, "terrain_raw.npy"),
                os.path.join(case_dir, "terrain_raw.npy"))

    dx = params["dx"]
    film_depth = params["film_depth_m"]
    total_time = params["total_time_s"]

    rc = run_batch.sh(f"python3 make_alpha.py {film_depth}", case_dir, log)
    if rc != 0:
        L("make_alpha FAILED"); return False

    cd_path = os.path.join(case_dir, "system", "controlDict")
    text = open(cd_path).read()
    text = re.sub(r"endTime\s+\d+(\.\d+)?;", f"endTime         {total_time:.0f};", text)
    open(cd_path, "w").write(text)

    rc = run_batch.sh("blockMesh", case_dir, log)
    if rc != 0:
        L("blockMesh FAILED"); return False

    rc = run_batch.sh(f"python3 warp_terrain_real.py {dx}", case_dir, log)
    if rc != 0:
        L("warp FAILED"); return False

    rc = run_batch.sh("decomposePar", case_dir, log)
    if rc != 0:
        L("decomposePar FAILED"); return False

    L(f"starting interFoam (total_time={total_time:.0f}s, rain={params['rain_mm']:.0f}mm)...")
    run_batch.sh("mpirun --allow-run-as-root --use-hwthread-cpus -np 8 interFoam -parallel", case_dir, log)

    rc = run_batch.sh("reconstructPar", case_dir, log)
    if rc != 0:
        L("reconstructPar FAILED"); return False

    time_dirs = [d for d in os.listdir(case_dir) if re.match(r"^\d+(\.\d+)?$", d) and d != "0"]
    if not time_dirs:
        L("no time directories found"); return False
    time_dirs.sort(key=lambda d: float(d))
    latest = time_dirs[-1]

    nx = ny = params["size"]
    dz = run_batch.BOX_H / run_batch.NZ
    depth_series, times = [], []
    for t in time_dirs:
        alpha_path = os.path.join(case_dir, t, "alpha.water")
        if not os.path.exists(alpha_path):
            continue
        alpha = run_batch.read_alpha_field(alpha_path, nx, ny, run_batch.NZ)
        depth_series.append((alpha * dz).sum(axis=0).astype(np.float32))
        times.append(float(t))
    if not depth_series:
        L("alpha.water missing at all time steps"); return False

    depth_series = np.stack(depth_series, axis=0)
    times = np.array(times, dtype=np.float32)
    depth = depth_series[-1]

    terrain = np.load(os.path.join(case_dir, "terrain_raw.npy"))
    out_dir = os.path.join(DATASET, name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "terrain.npy"), terrain)
    np.save(os.path.join(out_dir, "depth.npy"), depth)
    np.save(os.path.join(out_dir, "depth_series.npy"), depth_series)
    np.save(os.path.join(out_dir, "times.npy"), times)

    meta = dict(params)
    meta["source_terrain_sample"] = source_name
    meta["actual_end_time"] = latest
    meta["n_time_steps"] = int(len(times))
    meta["max_depth"] = float(depth.max())
    meta["mean_depth"] = float(depth.mean())
    meta["wall_clock_s"] = time.time() - t0
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    L(f"DONE in {meta['wall_clock_s']:.0f}s  max_depth={meta['max_depth']:.4f}m "
      f"mean_depth={meta['mean_depth']:.4f}m")
    shutil.rmtree(case_dir)
    return True


def main(n_source, n_variants, seed, start_idx):
    os.makedirs(DATASET, exist_ok=True)
    os.makedirs(run_batch.WORK, exist_ok=True)
    index = json.load(open(os.path.join(STAGING, "index.json"), encoding="utf-8"))
    targets = index[start_idx:start_idx + n_source]
    rng = np.random.default_rng(seed)

    total = len(targets) * n_variants
    done = 0
    n_ok = 0
    for base_params in targets:
        base_name = base_params["sample"]
        for v in range(n_variants):
            done += 1
            new_rain_mm = float(rng.uniform(20.0, 80.0))
            tries = 0
            while abs(new_rain_mm - base_params["rain_mm"]) < 15.0 and tries < 20:
                new_rain_mm = float(rng.uniform(20.0, 80.0))
                tries += 1

            new_params = dict(base_params)
            variant_name = f"{base_name}_rv{v + 1}"
            new_params["sample"] = variant_name
            new_params["rain_mm"] = new_rain_mm
            new_params["film_depth_m"] = new_rain_mm / 1000.0

            print(f"[{done}/{total}] {variant_name} (원본 지형: {base_name}, rain {new_rain_mm:.0f}mm)")
            ok = run_one_variant(new_params, base_name, done, total)
            if ok:
                n_ok += 1

    print(f"\n강우 시나리오 추가 배치 완료: {n_ok}/{total} 성공")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_source", type=int, default=20)
    p.add_argument("--n_variants", type=int, default=2)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--start_idx", type=int, default=0)
    args = p.parse_args()
    main(args.n_source, args.n_variants, args.seed, args.start_idx)
