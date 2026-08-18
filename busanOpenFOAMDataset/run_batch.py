#!/usr/bin/env python3
"""
staging/sample_XXXX (terrain_raw.npy + params.json)를 하나씩 읽어서
실제 OpenFOAM(interFoam, 8-way 병렬) 케이스를 만들고 돌린 뒤,
최종 시간의 alpha.water 필드를 수심맵으로 변환해 dataset/에 저장한다.
WSL 안에서 실행해야 함 (OpenFOAM 경로 공백 제약 때문에 홈 디렉토리에서 작업).
"""

import os
import re
import json
import shutil
import subprocess
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
STAGING = os.path.join(BASE, "staging")
TEMPLATE = os.path.join(BASE, "caseTemplate")
DATASET = os.path.join(BASE, "dataset")
WORK = os.path.join(BASE, "work")  # 케이스 실행용 임시 폴더

FOAM_SRC = "source /usr/lib/openfoam/openfoam2212/etc/bashrc"
NZ = 10
BOX_H = 1.0  # m


def sh(cmd, cwd, log):
    full = f"{FOAM_SRC} && {cmd}"
    with open(log, "a") as f:
        f.write(f"\n$ {cmd}\n")
    result = subprocess.run(["bash", "-lc", full], cwd=cwd, capture_output=True, text=True)
    with open(log, "a") as f:
        f.write(result.stdout)
        f.write(result.stderr)
    return result.returncode


def read_alpha_field(path, nx, ny, nz):
    text = open(path).read()
    m = re.search(r"internalField\s+nonuniform List<scalar>\s*\n(\d+)\n\((.*?)\)\s*;", text, re.DOTALL)
    n = int(m.group(1))
    vals = np.array([float(x) for x in m.group(2).split()])
    assert len(vals) == n, (len(vals), n)
    return vals.reshape(nz, ny, nx)


def run_one(params, idx, n_total):
    name = params["sample"]
    case_dir = os.path.join(WORK, name)
    log = os.path.join(BASE, "batch_log.txt")

    def L(msg):
        line = f"[{idx+1}/{n_total}] {name}: {msg}"
        print(line, flush=True)
        with open(log, "a") as f:
            f.write(line + "\n")

    t0 = time.time()

    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    shutil.copytree(TEMPLATE, case_dir)
    shutil.copy(os.path.join(STAGING, name, "terrain_raw.npy"), os.path.join(case_dir, "terrain_raw.npy"))

    dx = params["dx"]
    film_depth = params["film_depth_m"]
    total_time = params["total_time_s"]

    rc = sh(f"python3 make_alpha.py {film_depth}", case_dir, log)
    if rc != 0:
        L("make_alpha FAILED"); return False

    # controlDict endTime 갱신
    cd_path = os.path.join(case_dir, "system", "controlDict")
    text = open(cd_path).read()
    text = re.sub(r"endTime\s+\d+(\.\d+)?;", f"endTime         {total_time:.0f};", text)
    open(cd_path, "w").write(text)

    rc = sh("blockMesh", case_dir, log)
    if rc != 0:
        L("blockMesh FAILED"); return False

    rc = sh(f"python3 warp_terrain_real.py {dx}", case_dir, log)
    if rc != 0:
        L("warp FAILED"); return False

    rc = sh("decomposePar", case_dir, log)
    if rc != 0:
        L("decomposePar FAILED"); return False

    L(f"starting interFoam (total_time={total_time:.0f}s)...")
    rc = sh("mpirun --allow-run-as-root --use-hwthread-cpus -np 8 interFoam -parallel", case_dir, log)
    if rc != 0:
        L("interFoam FAILED (non-zero exit, check batch_log.txt)")
        # 그래도 마지막 타임스텝까지는 결과가 있을 수 있으니 계속 진행

    # -latestTime이 아니라 전체 시간을 reconstruct. controlDict가 writeInterval=2s로
    # 이미 다 쓰고 있던 중간 스텝들을 (지금까지는) 버리고 있었으므로, 배수 경로/시간별
    # 배수맵 기능을 위해 전체 시간을 살려서 depth_series로 같이 저장한다.
    rc = sh("reconstructPar", case_dir, log)
    if rc != 0:
        L("reconstructPar FAILED"); return False

    # 시간 디렉토리 전체(오름차순)
    time_dirs = [d for d in os.listdir(case_dir)
                 if re.match(r"^\d+(\.\d+)?$", d) and d != "0"]
    if not time_dirs:
        L("no time directories found after run"); return False
    time_dirs.sort(key=lambda d: float(d))
    latest = time_dirs[-1]

    nx = ny = params["size"]
    dz = BOX_H / NZ

    depth_series = []
    times = []
    for t in time_dirs:
        alpha_path = os.path.join(case_dir, t, "alpha.water")
        if not os.path.exists(alpha_path):
            continue
        alpha = read_alpha_field(alpha_path, nx, ny, NZ)
        depth_series.append((alpha * dz).sum(axis=0).astype(np.float32))
        times.append(float(t))

    if not depth_series:
        L(f"alpha.water missing at all time steps"); return False

    depth_series = np.stack(depth_series, axis=0)  # (T, ny, nx)
    times = np.array(times, dtype=np.float32)
    depth = depth_series[-1]  # 기존 파이프라인과 호환되는 최종 시점 스냅샷

    terrain = np.load(os.path.join(case_dir, "terrain_raw.npy"))

    out_dir = os.path.join(DATASET, name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "terrain.npy"), terrain)
    np.save(os.path.join(out_dir, "depth.npy"), depth)
    np.save(os.path.join(out_dir, "depth_series.npy"), depth_series)
    np.save(os.path.join(out_dir, "times.npy"), times)
    meta = dict(params)
    meta["actual_end_time"] = latest
    meta["n_time_steps"] = int(len(times))
    meta["max_depth"] = float(depth.max())
    meta["mean_depth"] = float(depth.mean())
    meta["wall_clock_s"] = time.time() - t0
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    L(f"DONE in {meta['wall_clock_s']:.0f}s  t={latest} ({meta['n_time_steps']} steps)  "
      f"max_depth={meta['max_depth']:.4f}m mean_depth={meta['mean_depth']:.4f}m")

    shutil.rmtree(case_dir)
    return True


def main():
    os.makedirs(DATASET, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    index = json.load(open(os.path.join(STAGING, "index.json"), encoding="utf-8"))

    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_ok = 0
    for i, params in enumerate(index):
        if i < start_idx:
            continue
        ok = run_one(params, i, len(index))
        if ok:
            n_ok += 1
    print(f"\n배치 완료: {n_ok}/{len(index) - start_idx} 성공")


if __name__ == "__main__":
    main()
