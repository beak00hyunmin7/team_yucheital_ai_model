"""
예측 배수맵을 '실제 등고선(m 라벨) + 음영기복 + 축척/방위 + 배수구 후보' 형태의
지도 스타일 그림으로 렌더링한다. ai_model/scripts/visualize_inference.py 의
make_overlay_figure 를 웹 백엔드에서 재사용하도록 옮긴 것.

지형 고도격자(terrain)가 있어야 등고선 라벨/음영기복/축척이 가능하므로
'등고선 데이터(terrain)' 모드에서만 사용한다. '등고선 이미지' 모드는 원본 고도값이
없어 이 그림을 만들 수 없다.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LightSource, PowerNorm  # noqa: E402
from matplotlib.patches import FancyArrow  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

matplotlib.rcParams["font.family"] = [
    "Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.facecolor"] = "white"

UPSCALE = 4


def _up(arr, order=3):
    return zoom(arr, UPSCALE, order=order)


def _scale_bar(ax, dx, n, length_m=200):
    bar = length_m / dx * UPSCALE
    x0, y0 = n * 0.05, n * 0.04
    ax.plot([x0, x0 + bar], [y0, y0], color="black", lw=3, solid_capstyle="butt", zorder=6)
    for xx in (x0, x0 + bar):
        ax.plot([xx, xx], [y0 - n * 0.008, y0 + n * 0.008], color="black", lw=2, zorder=6)
    ax.text(x0 + bar / 2, y0 + n * 0.02, f"{length_m}m", ha="center", fontsize=9,
            fontweight="bold", zorder=6)


def _north(ax, n):
    x0, y0 = n * 0.07, n * 0.85
    ax.add_patch(FancyArrow(x0, y0, 0, n * 0.08, width=n * 0.006,
                            head_width=n * 0.025, head_length=n * 0.03, color="black", zorder=6))
    ax.text(x0, y0 + n * 0.11, "N", ha="center", fontsize=11, fontweight="bold", zorder=6)


def render_analysis_figure(
    terrain: np.ndarray,
    intensity: np.ndarray,
    dx: float,
    *,
    title: str = "AI 예측 배수맵",
    subtitle: str | None = None,
    cbar_label: str = "상대적 침수 강도 (밝을수록 안전, 진할수록 위험)",
    mark_idx: tuple[int, int] | None = None,
) -> bytes:
    """terrain, intensity: 같은 shape (H, W). mark_idx: (row, col) 배수구 후보."""
    bg = _up(intensity.astype(float))
    terr = _up(terrain.astype(float))
    n = bg.shape[0]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ls = LightSource(azdeg=315, altdeg=45)
    ax.imshow(ls.hillshade(terr, vert_exag=1.5), origin="lower", cmap="gray",
              vmin=0.55, vmax=1.0, zorder=0)
    norm = PowerNorm(gamma=0.6, vmin=0.0, vmax=max(float(bg.max()), 1e-6))
    im = ax.imshow(bg, origin="lower", cmap="Blues", norm=norm, alpha=0.92,
                   interpolation="bilinear", zorder=1)
    lo, hi = float(np.percentile(terr, 2)), float(np.percentile(terr, 98))
    levels = np.linspace(lo, hi, 11)
    cs = ax.contour(terr, levels=levels, colors="#3a3a3a", linewidths=0.7, alpha=0.85,
                    origin="lower", zorder=2)
    ax.clabel(cs, cs.levels[::2], inline=True, fontsize=7, fmt="%.0fm")

    if mark_idx is not None:
        my, mx = mark_idx[0] * UPSCALE, mark_idx[1] * UPSCALE
        ax.scatter([mx], [my], c="red", s=220, marker="*", edgecolors="white",
                   linewidths=1.2, zorder=6)
        # 라벨은 항상 중앙 쪽으로 오프셋 (모서리에서 잘리거나 방위표와 겹치지 않게)
        ox = -n * 0.16 if mx > n * 0.5 else n * 0.16
        oy = -n * 0.13 if my > n * 0.5 else n * 0.13   # 위쪽 별이면 라벨은 아래로
        ha = "right" if mx > n * 0.5 else "left"
        ax.annotate("배수구 설치 후보", xy=(mx, my), xytext=(mx + ox, my + oy),
                    color="red", fontsize=11, fontweight="bold", zorder=7, ha=ha,
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    _scale_bar(ax, dx, n)
    _north(ax, n)
    ax.set_title(title if not subtitle else f"{title}\n{subtitle}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(cbar_label, fontsize=10)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def intensity_from_prediction(pred_img, out_shape: tuple[int, int]) -> np.ndarray:
    """예측 배수맵 PIL 이미지 -> (H, W) 상대 침수 강도 (255 - 밝기평균, 클수록 위험)."""
    from PIL import Image

    arr = np.asarray(pred_img.convert("RGB").resize((out_shape[1], out_shape[0]),
                                                    resample=Image.BICUBIC)).astype(float)
    return np.clip(255.0 - arr.mean(axis=2), 0.0, None)
