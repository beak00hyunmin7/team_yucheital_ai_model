"""
고도(elevation)/수심(depth) 2D 배열 -> RGB 이미지 렌더링 공용 함수.
scripts/render_from_openfoam.py와 scripts/generate_dataset.py가 함께 쓴다.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_contour_rgb(X, Y, Z, levels=20, figsize=(4, 4), dpi=64):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.contourf(X, Y, Z, levels=levels, cmap="terrain")
    ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.3)
    ax.set_axis_off()
    ax.set_aspect("equal")
    return _fig_to_rgb(fig)


def render_depth_rgb(depth, vmax=None, figsize=(4, 4), dpi=64):
    if vmax is None:
        vmax = max(float(depth.max()), 1e-6)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(depth, origin="lower", cmap="Blues", vmin=0.0, vmax=vmax)
    ax.set_axis_off()
    return _fig_to_rgb(fig)


def _fig_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = buf[:, :, :3].copy()
    plt.close(fig)
    return rgb
