"""
2D 지표 유출(overland flow) 간이 시뮬레이터.

실제 OpenFOAM/얕은물방정식(shallow water eq.)을 매 스텝 그대로 풀지 않고,
"물은 수위(지형고도+수심)가 높은 곳에서 낮은 곳으로, 경사가 클수록/수심이
깊을수록 빨리 이동한다"는 물리적 직관을 셀룰러 오토마타(CA) 방식으로
근사한 확산파(diffusive wave) 모델이다.

핵심 성질:
- 각 셀에서 한 스텝에 나가는 물의 양은 그 셀이 보유한 수심을 절대 넘지 않음
  (outflow_frac을 0~1로 clip) -> 질량 보존이 항상 성립하고 수심이 음수가 될 수 없음
  (안정성 걱정 없이 그대로 배치 실행 가능).
- 도메인 경계 밖은 "같은 고도의 마른 땅"으로 취급 -> 경계에 닿은 물은 자연 배수(유출)됨.

이 모델은 학습 데이터를 빠르게 대량 생성하기 위한 1차 근사이고,
추후 OpenFOAM 결과와 비교해 파라미터(K, n 지수 등)를 보정하면 됨.
"""

import numpy as np


def simulate_overland_flow(
    z,
    dx=1.0,
    dt=1.0,
    rain_rate=30.0 / 1000 / 3600,   # 30 mm/hr -> m/s
    rain_duration=1800.0,           # 강우 지속시간 [s]
    total_time=5400.0,              # 전체 시뮬레이션 시간(강우+배수) [s]
    k_rate=0.4,                     # 확산(배수) 속도 계수, 클수록 빨리 빠짐
    return_history=False,
    history_every=None,
):
    """
    z: (ny, nx) 지형 고도 배열 [m]
    반환: h (ny, nx) 최종 수심 배열 [m] (return_history=True면 (h, history) 튜플)
    """
    ny, nx = z.shape
    h = np.zeros_like(z, dtype=np.float64)
    z = z.astype(np.float64)

    n_steps = int(round(total_time / dt))
    history = [] if return_history else None

    for step in range(n_steps):
        t = step * dt
        if t < rain_duration:
            h += rain_rate * dt

        # 경계 밖 = 같은 고도의 마른 땅(open boundary, sink)
        z_pad = np.pad(z, 1, mode="edge")
        h_pad = np.pad(h, 1, mode="constant", constant_values=0.0)
        H_pad = z_pad + h_pad

        H_c = H_pad[1:-1, 1:-1]
        H_N = H_pad[:-2, 1:-1]
        H_S = H_pad[2:, 1:-1]
        H_E = H_pad[1:-1, 2:]
        H_W = H_pad[1:-1, :-2]

        sN = np.maximum((H_c - H_N) / dx, 0.0)
        sS = np.maximum((H_c - H_S) / dx, 0.0)
        sE = np.maximum((H_c - H_E) / dx, 0.0)
        sW = np.maximum((H_c - H_W) / dx, 0.0)

        # 수심이 깊을수록 더 빨리 흐름 (Manning 식의 h^(5/3) 의존성을 단순화한 h^(2/3) 가중치)
        w_h = np.power(np.maximum(h, 0.0), 2.0 / 3.0)
        wN, wS, wE, wW = sN * w_h, sS * w_h, sE * w_h, sW * w_h
        total_w = wN + wS + wE + wW
        total_w_safe = np.where(total_w > 0, total_w, 1.0)

        outflow_frac = np.minimum(1.0, k_rate * total_w)  # 항상 0~1 -> 질량 안전
        outflow_amount = h * outflow_frac

        fN = outflow_amount * (wN / total_w_safe)
        fS = outflow_amount * (wS / total_w_safe)
        fE = outflow_amount * (wE / total_w_safe)
        fW = outflow_amount * (wW / total_w_safe)

        h -= outflow_amount
        h[:-1, :] += fN[1:, :]
        h[1:, :] += fS[:-1, :]
        h[:, 1:] += fE[:, :-1]
        h[:, :-1] += fW[:, 1:]

        if return_history and history_every and step % history_every == 0:
            history.append(h.copy())

    if return_history:
        return h, history
    return h


def mass_balance_check(z, h_final, dx, rain_rate, rain_duration):
    """디버깅용: 투입된 비 부피 대비 (남은 수량 + 유출된 수량) 확인"""
    domain_area = z.size * dx * dx
    rain_volume_in = rain_rate * rain_duration * domain_area
    remaining_volume = h_final.sum() * dx * dx
    outflow_volume = rain_volume_in - remaining_volume
    return {
        "rain_volume_in_m3": rain_volume_in,
        "remaining_volume_m3": remaining_volume,
        "outflow_volume_m3": outflow_volume,
        "outflow_ratio": outflow_volume / rain_volume_in if rain_volume_in > 0 else None,
    }


if __name__ == "__main__":
    from generate_terrain import generate_hill_terrain
    import matplotlib.pyplot as plt

    z, X, Y = generate_hill_terrain(size=128, dx=1.0, seed=0)
    h = simulate_overland_flow(z, dx=1.0, dt=1.0, rain_duration=1800, total_time=5400)

    stats = mass_balance_check(z, h, dx=1.0, rain_rate=30.0 / 1000 / 3600, rain_duration=1800)
    print(stats)
    print("max depth [m]:", h.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    c0 = axes[0].contourf(X, Y, z, levels=20, cmap="terrain")
    axes[0].set_title("Terrain (input)")
    fig.colorbar(c0, ax=axes[0], label="Elevation [m]")

    c1 = axes[1].imshow(h, origin="lower", cmap="Blues", extent=[X.min(), X.max(), Y.min(), Y.max()])
    axes[1].set_title("Water depth (simulated output)")
    fig.colorbar(c1, ax=axes[1], label="Depth [m]")

    for ax in axes:
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig("flow_preview.png", dpi=120)
    print("saved flow_preview.png")
