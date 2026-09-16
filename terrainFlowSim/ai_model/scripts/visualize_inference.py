"""
등고선 -> AI 예측 배수맵을 등고선(실제 고도, m) 오버레이와 함께 시각화한다.
정답 depth.npy가 있으면 같은 방식으로 정답 이미지도 따로 만들어서 비교할 수 있다.

사용법 (ai_model/ 디렉터리에서):
    # 지형 원시 데이터로 실행 (등고선 이미지는 자동 렌더링)
    python scripts/visualize_inference.py --terrain path/to/terrain.npy

    # 정답(depth.npy)까지 있으면 같이 비교
    python scripts/visualize_inference.py --terrain terrain.npy --depth depth.npy

    # 이미 렌더링된 등고선 이미지를 직접 넣고 싶으면
    python scripts/visualize_inference.py --terrain terrain.npy --contour_image data/contours/xxx.png

    # 체크포인트 지정 (기본은 AI_MODEL_CHECKPOINT 환경변수 또는 checkpoints/best.pt)
    python scripts/visualize_inference.py --terrain terrain.npy --checkpoint checkpoints_gan/best.pt
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, PowerNorm
from matplotlib.patches import FancyArrow
from scipy.ndimage import zoom
from PIL import Image

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.facecolor"] = "white"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

UPSCALE = 4  # 저해상도(96x96 등) 배열을 부드럽게 보이도록 확대하는 배수


def _smooth_upsample(arr, factor=UPSCALE, order=3):
    """저해상도 격자를 매끈하게 보간해서 계단 현상을 없앤다 (bicubic)."""
    return zoom(arr, factor, order=order)


def _add_scale_bar(ax, dx, n_cols, length_m=200):
    """지도 좌하단에 실제 거리(m) 기준 축척 막대를 그린다."""
    bar_px = length_m / dx
    x0 = n_cols * 0.05
    y0 = n_cols * 0.04
    ax.plot([x0, x0 + bar_px], [y0, y0], color="black", linewidth=3, solid_capstyle="butt", zorder=6)
    ax.plot([x0, x0], [y0 - n_cols * 0.008, y0 + n_cols * 0.008], color="black", linewidth=2, zorder=6)
    ax.plot([x0 + bar_px, x0 + bar_px], [y0 - n_cols * 0.008, y0 + n_cols * 0.008], color="black", linewidth=2, zorder=6)
    ax.text(x0 + bar_px / 2, y0 + n_cols * 0.02, f"{length_m}m", ha="center", fontsize=9,
            fontweight="bold", color="black", zorder=6)


def _add_north_arrow(ax, n_cols):
    x0 = n_cols * 0.92
    y0 = n_cols * 0.85
    ax.add_patch(FancyArrow(x0, y0, 0, n_cols * 0.08, width=n_cols * 0.006,
                             head_width=n_cols * 0.025, head_length=n_cols * 0.03,
                             color="black", zorder=6))
    ax.text(x0, y0 + n_cols * 0.11, "N", ha="center", fontsize=11, fontweight="bold", zorder=6)


def make_overlay_figure(background, terrain, cmap, vmin, vmax, cbar_label, title,
                         out_path, dx=10.416666666666666, mark_max=False, max_idx=None,
                         subtitle=None):
    # 저해상도 배열을 부드럽게 확대(bicubic) - 등고선/침수 강도 모두 계단현상 없이 매끈하게
    bg_hi = _smooth_upsample(background)
    terrain_hi = _smooth_upsample(terrain)
    n = bg_hi.shape[0]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # 음영기복(hillshade)으로 지형에 입체감을 준 뒤 그 위에 침수 강도를 겹친다.
    # 배수맵이 흐려 보이는 문제 해결: (1) 지형 음영 자체의 명암 대비를 좁혀서
    # 옅게 깔고(vmin을 0.2->0.55로), (2) 침수 강도 레이어의 불투명도를 0.75->0.92로
    # 올려서 회색이 섞여 탁해지는 걸 줄이고, (3) PowerNorm(gamma<1)으로 저수심 구간도
    # 흰색에 묻히지 않고 옅은 파랑으로 보이게 대비를 살렸다.
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(terrain_hi, vert_exag=1.5)
    ax.imshow(hillshade, origin="lower", cmap="gray", vmin=0.55, vmax=1.0, zorder=0)

    norm = PowerNorm(gamma=0.6, vmin=vmin, vmax=max(vmax, 1e-6))
    im = ax.imshow(bg_hi, origin="lower", cmap=cmap, norm=norm,
                    alpha=0.92, interpolation="bilinear", zorder=1)

    cs = ax.contour(terrain_hi, levels=10, colors="#3a3a3a", linewidths=0.8, alpha=0.9,
                     origin="lower", zorder=2)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0fm")

    if mark_max and max_idx is not None:
        my, mx = max_idx[0] * UPSCALE, max_idx[1] * UPSCALE
        ax.scatter([mx], [my], c="red", s=220, marker="*", edgecolors="white",
                   linewidths=1.2, zorder=6)
        ax.annotate("배수구 설치 후보", xy=(mx, my), xytext=(mx + n * 0.05, my - n * 0.10),
                    color="red", fontsize=11, fontweight="bold", zorder=6,
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    _add_scale_bar(ax, dx, n)
    _add_north_arrow(ax, n)

    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full_title, fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(cbar_label, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)
    print("saved:", out_path)


def main(args):
    if args.checkpoint:
        os.environ["AI_MODEL_CHECKPOINT"] = args.checkpoint

    from src.inference_api import (predict_from_image, predict_from_terrain,
                                    predict_depth_from_image, predict_depth_from_terrain,
                                    current_checkpoint, current_cond_dim, supports_depth_output)
    from src.utils.render_fields import render_contour_rgb

    terrain = np.load(args.terrain)
    ny, nx = terrain.shape

    # 조건(강우량/시점)이 필요한 체크포인트(v5)면 CLI 인자를 그대로 넘긴다.
    cond_kw = {}
    if current_cond_dim() >= 1:
        cond_kw["rain_mm"] = args.rain_mm
    if current_cond_dim() >= 2:
        cond_kw["time_s"] = args.time_s

    # v5 이후 체크포인트는 예측을 실제 수심[m]으로 환산할 수 있어서, 정답과
    # 같은 물리 단위/같은 색 스케일로 비교할 수 있다.
    depth_pred = None
    if supports_depth_output():
        if args.contour_image:
            depth_pred = predict_depth_from_image(args.contour_image, out_shape=terrain.shape, **cond_kw)
        else:
            depth_pred = predict_depth_from_terrain(terrain, dx=args.dx, **cond_kw)
        pred_img = None
    elif args.contour_image:
        pred_img = predict_from_image(args.contour_image, **cond_kw)
    else:
        pred_img = predict_from_terrain(terrain, dx=args.dx, **cond_kw)

    if depth_pred is not None:
        intensity = depth_pred            # 실제 수심[m]
        intensity_label = "예측 침수 깊이 [m]"
    else:
        pred_arr = np.array(pred_img.resize((nx, ny), resample=Image.BICUBIC)).astype(float)
        intensity = np.clip(255.0 - pred_arr.mean(axis=2), 0, None)
        intensity_label = "상대적 침수 강도 (밝을수록 안전, 진할수록 위험)"
    # render_contour_rgb 가 등고선 이미지를 상하반전(matplotlib Y축)으로 만들기 때문에
    # 이미지 파이프라인을 통과한 예측도 terrain 배열 기준으로는 상하반전 상태다.
    # terrain(등고선)과 겹쳐 그리려면 다시 뒤집어 맞춘다.
    intensity = np.flipud(intensity)
    max_idx = np.unravel_index(np.argmax(intensity), intensity.shape)

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.name or os.path.splitext(os.path.basename(args.terrain))[0]

    make_overlay_figure(
        intensity, terrain, "Blues", 0, intensity.max(),
        intensity_label,
        "AI 예측 배수맵",
        os.path.join(args.out_dir, f"{name}_overlay_prediction.png"),
        dx=args.dx, mark_max=True, max_idx=max_idx,
        subtitle=f"지형 고저차 {terrain.max() - terrain.min():.1f}m · 회색 음영 = 실제 지형 굴곡",
    )

    if args.depth:
        depth_gt = np.load(args.depth)
        make_overlay_figure(
            depth_gt, terrain, "Blues", 0, depth_gt.max(),
            "침수 깊이 [m]",
            "실제 시뮬레이션 정답 (OpenFOAM)",
            os.path.join(args.out_dir, f"{name}_overlay_groundtruth.png"),
            dx=args.dx, mark_max=True,
            max_idx=np.unravel_index(np.argmax(depth_gt), depth_gt.shape),
            subtitle=f"최대 침수 깊이 {depth_gt.max():.3f}m",
        )

    print(f"사용한 체크포인트: {current_checkpoint()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", type=str, required=True, help="지형 고도 배열(.npy) 경로")
    parser.add_argument("--depth", type=str, default=None, help="정답 침수 깊이 배열(.npy) 경로 (선택)")
    parser.add_argument("--contour_image", type=str, default=None,
                         help="이미 렌더링된 등고선 이미지 경로 (없으면 --terrain에서 자동 렌더링)")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="모델 체크포인트 경로 (기본: AI_MODEL_CHECKPOINT 환경변수 또는 checkpoints/best.pt)")
    parser.add_argument("--dx", type=float, default=10.416666666666666, help="지형 셀 크기 [m]")
    parser.add_argument("--rain_mm", type=float, default=45.0,
                         help="강우량 [mm] (강우 조건 체크포인트에서만 사용, 학습 범위 20~80)")
    parser.add_argument("--time_s", type=float, default=120.0,
                         help="강우 시작 후 경과 시간 [s] (시점 조건 체크포인트에서만 사용, 0~120)")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--name", type=str, default=None, help="출력 파일명 접두사 (기본: terrain 파일명)")
    args = parser.parse_args()

    main(args)
