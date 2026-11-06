"""
학습 입력에 붙일 "고도 채널"을 만든다.

왜 필요한가 (scripts/diagnose_input_info.py 결과):
  학습 입력인 등고선 그림은 contourf(levels=20) 으로 고도를 20단계로 양자화한다.
  평균 고저차 31.5m 기준 한 칸이 1.58m인데, 배수 물길이 파인 깊이는 보통 그보다 얕다.
  실제로 원본 고도와 20단계 양자화 고도의 D8 흐름누적을 비교하면 상관 0.494,
  **물길 위치 일치 IoU 는 0.216** 에 그친다. 즉 물길의 78%가 입력에서 제자리를 잃는다.

  기울기(고주파) 손실을 넣어도 번짐(sigma≈2.2)이 전혀 줄지 않은 이유가 이것이다.
  없는 정보를 그려내라고 시킨 셈이었다.

무엇을 만드나:
  data/elev/{region}_{sample}_{aug}.npy  — float32, [0,1] 정규화된 고도 격자.
  시점(_t0..)에 따라 지형이 변하지 않으므로 시점 접미사는 붙이지 않는다
  (파일 수를 6배 줄인다). 데이터셋이 이미지 이름에서 _t 를 떼고 찾아 쓴다.

정규화: 샘플별 min-max. 등고선 렌더도 샘플마다 색을 다시 맞추므로 같은 성질이고,
배수 경로는 절대 고도가 아니라 상대적 고저에 의해 정해진다.

사용법 (ai_model/ 에서):
    .venv\\Scripts\\python.exe scripts\\build_elev_channel.py
"""

import os
import re
import sys
import glob
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from prepare_openfoam_dataset import REGIONS, d4_transforms  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "elev")
D4 = d4_transforms()


def normalize(z):
    """샘플별 min-max -> [0,1]. 평지(고저차 0)는 0.5 로 채운다."""
    lo, hi = float(z.min()), float(z.max())
    if hi - lo < 1e-9:
        return np.full_like(z, 0.5, dtype=np.float32)
    return ((z - lo) / (hi - lo)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="이미 있는 파일도 다시 만든다")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    made = skipped = 0

    for region, ddir in REGIONS.items():
        dirs = sorted(glob.glob(os.path.join(ddir, "sample_*")))
        print(f"[{region}] 샘플 {len(dirs)}개")
        for i, sdir in enumerate(dirs):
            sample = os.path.basename(sdir)
            tpath = os.path.join(sdir, "terrain.npy")
            if not os.path.exists(tpath):
                continue
            z = None
            for aug_name, fn in D4:
                out = os.path.join(OUT_DIR, f"{region}_{sample}_{aug_name}.npy")
                if os.path.exists(out) and not args.force:
                    skipped += 1
                    continue
                if z is None:
                    z = normalize(np.load(tpath))
                np.save(out, np.ascontiguousarray(fn(z)))
                made += 1
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(dirs)}")

    print(f"\n생성 {made}개, 건너뜀 {skipped}개 -> {OUT_DIR}")
    files = glob.glob(os.path.join(OUT_DIR, "*.npy"))
    if files:
        a = np.load(files[0])
        total = sum(os.path.getsize(f) for f in files) / 1024 / 1024
        print(f"확인: {os.path.basename(files[0])} shape={a.shape} "
              f"범위 {a.min():.2f}~{a.max():.2f}, 전체 {total:.0f} MB ({len(files)}개)")


def elev_key(image_name):
    """'seoul_sample_0012_rv1_r0_t3.png' -> 'seoul_sample_0012_rv1_r0' (시점 접미사 제거)."""
    return re.sub(r"_t\d+$", "", os.path.splitext(os.path.basename(image_name))[0])


if __name__ == "__main__":
    main()
