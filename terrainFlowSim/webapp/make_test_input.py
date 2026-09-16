"""
웹앱 테스트용 가상 지형 입력 파일 생성기.

두 입력 방식을 같은 조건에서 비교할 수 있도록, **하나의 지형**에서
  - 등고선 데이터 (고도 격자): .npy / .csv
  - 등고선 이미지: .png
를 같이 만든다. 이미지는 학습 데이터와 **똑같은 렌더러**(render_contour_rgb,
matplotlib terrain 20단계 + 검은 등고선)로 그린다. 이 스타일에서 멀어지면
정확도가 급락하므로 직접 그린 그림을 넣는 것보다 이쪽이 정상 동작 확인에 적합하다.

지형 규격은 학습 데이터(OpenFOAM)에 맞췄다:
  96x96 격자, dx 10.4167 m (=1km), 고도 60~103 m, 고저차 20 m 안팎.

사용법 (webapp/ 에서):
    ..\\ai_model\\.venv\\Scripts\\python.exe make_test_input.py
    ..\\ai_model\\.venv\\Scripts\\python.exe make_test_input.py --seed 7 --relief 35
"""

import os
import sys
import argparse

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

HERE = os.path.dirname(os.path.abspath(__file__))
AI_MODEL = os.path.abspath(os.path.join(HERE, "..", "ai_model"))
sys.path.insert(0, AI_MODEL)

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.utils.render_fields import render_contour_rgb  # noqa: E402

N = 96
DX = 10.416666666666666


def fractal_noise(n, rng, octaves=5):
    """저주파부터 고주파까지 겹쳐 쌓은 지형 노이즈 (실제 지형처럼 거칠기가 스케일마다 다름)."""
    out = np.zeros((n, n))
    for o in range(1, octaves + 1):
        size = 2 ** o + 1
        coarse = rng.normal(size=(size, size))
        up = zoom(coarse, n / size, order=3)[:n, :n]
        if up.shape != (n, n):  # zoom 이 반올림으로 1픽셀 어긋날 수 있다
            up = np.pad(up, ((0, n - up.shape[0]), (0, n - up.shape[1])), mode="edge")
        out += up / (2 ** o)
    return out


def carve_valleys(z, rng, strength=1.0):
    """가지 친 골짜기를 파서 물이 실제로 모일 수 있는 배수망을 만든다.

    노이즈만 쓰면 물이 갈 곳 없이 흩어져서 예측도 밋밋해진다. 본류 1개 + 지류 3개를
    꺾인 선으로 깔고, 선까지의 거리에 따라 고도를 낮춘다.
    """
    n = z.shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(float)

    # (본류, 지류들): 각각 꺾인 점들의 목록
    paths = [
        [(8, 14), (34, 40), (52, 55), (70, 74), (88, 90)],   # 본류 (좌상 -> 우하)
        [(10, 70), (30, 58), (48, 52)],                       # 지류 A
        [(78, 18), (62, 36), (54, 50)],                       # 지류 B
        [(40, 88), (52, 74), (64, 66)],                       # 지류 C
    ]
    widths = [7.0, 4.5, 4.5, 4.0]
    depths = [9.0, 5.0, 5.0, 4.0]

    carve = np.zeros_like(z)
    for path, w, d in zip(paths, widths, depths):
        dist = np.full_like(z, 1e9)
        for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
            # 선분까지의 거리
            vr, vc = r1 - r0, c1 - c0
            L2 = vr * vr + vc * vc
            t = np.clip(((yy - r0) * vr + (xx - c0) * vc) / L2, 0.0, 1.0)
            dist = np.minimum(dist, np.hypot(yy - (r0 + t * vr), xx - (c0 + t * vc)))
        carve += d * np.exp(-(dist ** 2) / (2 * w * w))

    jitter = gaussian_filter(rng.normal(size=z.shape), 3.0)
    return z - strength * carve * (1.0 + 0.15 * jitter / (jitter.std() + 1e-9))


def make_terrain(seed=20260916, base=68.0, relief=24.0):
    rng = np.random.default_rng(seed)
    z = fractal_noise(N, rng)
    z = gaussian_filter(z, 1.2)

    # 전체적인 경사 (물이 흐를 방향이 생기도록)
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    z += 0.35 * (yy / N) + 0.20 * (xx / N)

    z = (z - z.min()) / (z.max() - z.min() + 1e-12)
    z = base + z * relief
    z = carve_valleys(z, rng)
    z = gaussian_filter(z, 0.8)          # 골짜기 경계 매끄럽게
    return z.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--base", type=float, default=68.0, help="기준 고도 [m]")
    p.add_argument("--relief", type=float, default=24.0, help="고저차 [m]")
    p.add_argument("--out", default=os.path.join(HERE, "..", "..", "..", "테스트_입력"),
                   help="출력 폴더 (기본: 유체이탈/테스트_입력)")
    p.add_argument("--name", default="test_terrain")
    args = p.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    z = make_terrain(args.seed, args.base, args.relief)

    npy_path = os.path.join(out_dir, args.name + ".npy")
    csv_path = os.path.join(out_dir, args.name + ".csv")
    png_path = os.path.join(out_dir, args.name + "_contour.png")

    np.save(npy_path, z)
    np.savetxt(csv_path, z, delimiter=",", fmt="%.3f")

    X, Y = np.meshgrid(np.arange(N) * DX, np.arange(N) * DX)
    Image.fromarray(render_contour_rgb(X, Y, z)).save(png_path)

    print(f"지형: {z.shape[0]}x{z.shape[1]}, dx {DX:.4f} m "
          f"(가로세로 {N * DX:.0f} m)")
    print(f"고도 {z.min():.1f} ~ {z.max():.1f} m (고저차 {np.ptp(z):.1f} m, 표준편차 {z.std():.1f} m)")
    print("\n생성 파일:")
    for path in (npy_path, csv_path, png_path):
        print(f"  {os.path.basename(path):28s} {os.path.getsize(path) / 1024:8.1f} KB")
    print(f"\n위치: {out_dir}")
    print("\n사용법:")
    print("  · '등고선 데이터 (고도 격자)' 모드 -> .npy 또는 .csv")
    print("  · '등고선 이미지' 모드            -> _contour.png")
    print("  두 파일은 같은 지형이라 두 방식의 결과를 서로 비교할 수 있습니다.")


if __name__ == "__main__":
    main()
