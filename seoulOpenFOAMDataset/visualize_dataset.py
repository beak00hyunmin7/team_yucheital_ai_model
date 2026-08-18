"""
dataset/ 안의 (terrain.npy, depth.npy, meta.json) 샘플들을 시각화한다.
등고선(지형 contour)을 침수 깊이(depth) 맵 위에 겹쳐 그려서, 물이 어느
등고선 라인/지형 특징을 따라 고이는지 한눈에 볼 수 있게 한다.

사용법:
    python visualize_dataset.py --sample sample_0021          # 샘플 1개
    python visualize_dataset.py --top 6                       # max_depth 상위 N개
    python visualize_dataset.py --all                         # 전체 격자 요약(깊이맵만)
    python visualize_dataset.py --top 6 --out my_out_dir       # 출력 폴더 지정
"""

import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_DIR = os.path.join(HERE, "dataset")
DEFAULT_OUT_DIR = os.path.join(HERE, "viz")


def load_meta(dataset_dir):
    samples = sorted(d for d in os.listdir(dataset_dir) if d.startswith("sample_"))
    metas = []
    for s in samples:
        m = json.load(open(os.path.join(dataset_dir, s, "meta.json"), encoding="utf-8"))
        m["sample"] = s
        metas.append(m)
    return metas


def plot_contour_overlay(ax, terrain, depth, contour_levels=12, contour_color="dimgray"):
    """depth를 배경(Blues)으로, terrain 등고선을 위에 오버레이."""
    im = ax.imshow(depth, origin="lower", cmap="Blues", vmin=0)
    cs = ax.contour(
        terrain, levels=contour_levels, colors=contour_color,
        linewidths=0.6, alpha=0.8, origin="lower",
    )
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.0fm")
    ax.axis("off")
    return im


def render_sample(dataset_dir, out_dir, meta):
    s = meta["sample"]
    terrain = np.load(os.path.join(dataset_dir, s, "terrain.npy"))
    depth = np.load(os.path.join(dataset_dir, s, "depth.npy"))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = plot_contour_overlay(ax, terrain, depth)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="flood depth [m]")
    ax.set_title(
        f"{s}  (relief {meta['relief_m']:.1f}m, rain {meta['rain_mm']:.0f}mm, "
        f"max depth {meta['max_depth']:.3f}m)",
        fontsize=9,
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{s}_contour_overlay.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def render_grid(dataset_dir, out_dir, metas, ncols=6):
    nrows = int(np.ceil(len(metas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for i, m in enumerate(metas):
        r, c = divmod(i, ncols)
        s = m["sample"]
        terrain = np.load(os.path.join(dataset_dir, s, "terrain.npy"))
        depth = np.load(os.path.join(dataset_dir, s, "depth.npy"))
        plot_contour_overlay(axes[r, c], terrain, depth, contour_levels=6)
        axes[r, c].set_title(f"{s}\nmax {m['max_depth']:.3f}m", fontsize=7)

    for i in range(len(metas), nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis("off")

    fig.suptitle("Seoul dataset — flood depth with terrain contour overlay", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(out_dir, "all_contour_overlay_grid.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    p.add_argument("--out", default=DEFAULT_OUT_DIR)
    p.add_argument("--sample", help="특정 샘플 하나만 (예: sample_0021)")
    p.add_argument("--top", type=int, help="max_depth 상위 N개 개별 렌더링")
    p.add_argument("--all", action="store_true", help="전체 샘플을 격자 1장으로 요약")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    metas = load_meta(args.dataset_dir)

    if args.sample:
        target = next(m for m in metas if m["sample"] == args.sample)
        path = render_sample(args.dataset_dir, args.out, target)
        print("saved:", path)
    elif args.top:
        top_metas = sorted(metas, key=lambda m: -m["max_depth"])[: args.top]
        for m in top_metas:
            path = render_sample(args.dataset_dir, args.out, m)
            print("saved:", path)
    elif args.all:
        path = render_grid(args.dataset_dir, args.out, metas)
        print("saved:", path)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
