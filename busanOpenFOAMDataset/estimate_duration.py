"""
패치(유역)의 물리적 특성(크기, 평균 경사)에 따라 시뮬레이션 총 시간을 동적으로
정한다. 고정된 60초를 모든 패치에 똑같이 쓰는 건 방법론적으로 틀렸다는 지적을
반영 — 유역이 크거나 완만할수록 배수(도달)시간이 길어지므로 그만큼 더 오래
계산해야 한다.

물리적 근거: 경사면을 중력으로 미끄러져 내려가는 거리-시간 관계
    L = 1/2 * g*sin(theta) * t^2  ->  t = sqrt(2L / (g*S))
(S = 평균 경사, 작은 각도 근사로 sin(theta) ~= S)
이 값에 안전계수를 곱해서 "물이 거의 다 빠질 만큼 충분한" 시간을 추정한다.
바닥(최소)/천장(최대) 값으로 클리핑해서 극단적으로 완만하거나 급한 지형에서
비현실적인 값이 나오는 것을 막는다.
"""

import numpy as np

G = 9.81


def estimate_total_time(z, dx, safety_factor=3.0, t_min=20.0, t_max=120.0, slope_floor=0.01):
    """
    z: (ny, nx) 지형 고도 배열 [m]
    dx: 셀 크기 [m]
    반환: 권장 시뮬레이션 총 시간 [s]
    """
    ny, nx = z.shape
    L = max(ny, nx) * dx  # 유역 특성 길이 (도메인 한 변)

    gy, gx = np.gradient(z, dx)
    slope = np.sqrt(gx ** 2 + gy ** 2)
    mean_slope = max(float(slope.mean()), slope_floor)

    t_est = safety_factor * np.sqrt(2 * L / (G * mean_slope))
    return float(np.clip(t_est, t_min, t_max)), mean_slope, L


if __name__ == "__main__":
    import sys
    z = np.load(sys.argv[1])
    dx = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    t, s, L = estimate_total_time(z, dx)
    print(f"L={L:.0f}m mean_slope={s:.4f} -> recommended total_time={t:.1f}s")
