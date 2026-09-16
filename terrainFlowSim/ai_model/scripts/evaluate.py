"""
체크포인트 하나를 받아 val 셋 전체를 평가한다 (IMPROVEMENT_GUIDE.md Phase 0).

핵심 원칙 두 가지:

1. **배포되는 경로 그대로 잰다.**
   `등고선 PNG -> UNet -> 예측 RGB -> Blues 역LUT -> norm_to_depth -> 수심[m]`
   모델 출력 텐서를 직접 쓰면 실제 사용자가 받는 값이 아니라 중간값을 재게 된다.
   역LUT 양자화 오차까지 포함한 값이 진짜 성능이다.

2. **정답은 학습 타깃 PNG가 아니라 OpenFOAM 원본 depth(m)를 쓴다.**
   타깃 PNG와 비교하면 렌더링 손실이 정답 쪽에도 똑같이 들어가서 오차가 실제보다 작게 나온다.

사용법 (ai_model/ 에서):
    .venv\\Scripts\\python.exe scripts\\evaluate.py \
        --checkpoint checkpoints_v5/best.pt --config configs/config_v5_fixed_target.yaml --tag v5

    # 빠른 점검 (지형 20개만)
    .venv\\Scripts\\python.exe scripts\\evaluate.py --checkpoint ... --config ... --limit-terrains 20
"""

import os
import re
import sys
import csv
import json
import glob
import argparse
from collections import defaultdict

import numpy as np
import torch
import yaml
from PIL import Image

# Windows 콘솔 기본 인코딩(cp949)에서 한글/기호 출력이 깨지지 않도록 UTF-8로 고정한다.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from src.datasets.contour_flow_dataset import split_train_val, _group_key  # noqa: E402
from src.utils.render_fields import norm_to_depth, DEPTH_VMAX_M  # noqa: E402
from src.utils.image_utils import tensor_to_uint8  # noqa: E402
from prepare_openfoam_dataset import REGIONS, d4_transforms, _pick_timesteps  # noqa: E402

NAME_RE = re.compile(r"^(seoul|busan|gangwon)_(sample_\d+(?:_rv\d+)?)_(r[0-3]|f[0-3])_t(\d+)$")
D4 = dict(d4_transforms())

# 젖은영역 IoU 임계 [m]. 1cm = "발이 젖는다", 5cm = "차량 통행 지장".
WET_THRESHOLDS = [0.01, 0.05]


def to_array_orientation(depth_img):
    """예측 결과(이미지 방향) -> OpenFOAM terrain/depth 배열 방향.

    렌더링이 둘 다 matplotlib를 거치면서 세로축이 뒤집힌다:
      - render_contour_rgb: contourf는 Y가 위로 증가 -> 이미지 0행 = 배열 마지막 행
      - render_depth_rgb  : imshow(origin="lower") -> 동일하게 뒤집힘
    입력과 타깃이 같은 방향으로 뒤집혀 있어 학습에는 문제가 없지만, 예측 배열을
    원본 terrain/depth 배열과 비교하려면 되돌려야 한다. 실측으로도 확인됨
    (val 12장 평균 상관계수: 그대로 0.17 / flipud 0.79 / fliplr 0.15 / 180도 0.15).

    주의: inference_api.predict_depth_from_* 는 이 되돌림을 하지 않고 이미지 방향
    그대로 반환한다 (IMPROVEMENT_GUIDE.md Phase 3 참조).
    """
    return np.ascontiguousarray(np.flipud(depth_img))


# ---------------------------------------------------------------- 정답(GT) 로딩

class GroundTruth:
    """OpenFOAM 원본에서 (지형, 시점, 증강)에 해당하는 실제 수심[m]을 꺼내준다.

    같은 sample_dir의 t0..t5가 연속으로 조회되므로 마지막 하나만 캐시해도
    depth_series.npy 재로딩이 거의 사라진다.
    """

    def __init__(self):
        self._key = None
        self._steps = None
        self._meta = None

    def _load(self, region, sample_name):
        key = (region, sample_name)
        if key == self._key:
            return
        sdir = os.path.join(REGIONS[region], sample_name)
        meta_path = os.path.join(sdir, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        total_time = meta.get("total_time_s", 120.0)
        self._steps = _pick_timesteps(sdir, total_time)
        self._meta = meta
        self._key = key

    def get(self, region, sample_name, aug, t_idx):
        """(수심[m] 배열, dx[m], rain_mm, 실제시각[s]) 또는 None(원본 없음)."""
        sdir = os.path.join(REGIONS[region], sample_name)
        if not os.path.isdir(sdir):
            return None
        self._load(region, sample_name)
        if t_idx >= len(self._steps):
            return None
        depth, t_val = self._steps[t_idx]
        dx = self._meta.get("dx", 10.416666666666666)
        return np.ascontiguousarray(D4[aug](depth)), dx, self._meta.get("rain_mm"), t_val

    def terrain(self, region, sample_name, aug):
        sdir = os.path.join(REGIONS[region], sample_name)
        z = np.load(os.path.join(sdir, "terrain.npy"))
        return np.ascontiguousarray(D4[aug](z))


# ---------------------------------------------------------------- 추론 파이프라인

class Predictor:
    """inference_api와 동일한 전처리/후처리를 배치로 수행한다.

    inference_api는 한 장씩 처리하도록 돼 있어 7천 장 평가엔 너무 느리다.
    전처리(Resize+Normalize)와 역LUT를 GPU에서 배치로 돌리되, 연산 순서와
    상수는 inference_api._run / _image_to_depth와 같게 유지한다.
    """

    def __init__(self, checkpoint, image_size):
        os.environ["AI_MODEL_IMAGE_SIZE"] = str(image_size)
        from src import inference_api  # 환경변수 반영을 위해 늦게 import
        self.api = inference_api
        self.net = inference_api._load_model(checkpoint)
        self.device = inference_api.DEVICE
        self.image_size = image_size
        self.cond_dim = inference_api.current_cond_dim()
        self.in_channels = inference_api.current_in_channels()
        self.lut = torch.from_numpy(inference_api._LUT).to(self.device)  # (256, 3)

    def _to_input(self, path):
        img = Image.open(path).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1)
        return (t - 0.5) / 0.5

    def cond(self, rain_mm, time_s):
        if self.cond_dim == 0:
            return None
        parts = [self.api.normalize_rain(rain_mm)]
        if self.cond_dim == 2:
            parts.append(self.api.normalize_time(time_s))
        return torch.tensor(parts, dtype=torch.float32)

    @torch.no_grad()
    def forward(self, paths, conds):
        """(예측 텐서 [-1,1] (B,3,H,W), 입력 텐서) 반환."""
        x = torch.stack([self._to_input(p) for p in paths]).to(self.device)
        c = None
        if conds[0] is not None:
            c = torch.stack(conds).to(self.device)
        return self.net(x, c)

    @torch.no_grad()
    def to_depth(self, pred_t, out_hw):
        """예측 텐서 -> 수심[m] 배열 목록. 배포 경로(uint8 RGB -> resize -> 역LUT)와 동일."""
        out = []
        for i in range(pred_t.shape[0]):
            rgb = tensor_to_uint8(pred_t[i])                       # (H, W, 3) uint8
            img = Image.fromarray(rgb).resize((out_hw[1], out_hw[0]), Image.BICUBIC)
            arr = torch.from_numpy(np.asarray(img).astype(np.float32)).to(self.device)
            d2 = ((arr.reshape(-1, 1, 3) - self.lut.reshape(1, -1, 3)) ** 2).sum(-1)
            idx = d2.argmin(1).reshape(out_hw).float() / (self.lut.shape[0] - 1)
            out.append(norm_to_depth(idx.cpu().numpy()))
        return out


# ---------------------------------------------------------------- 지표

def pearson(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else float("nan")


def wet_iou(pred, gt, thr):
    p, g = pred >= thr, gt >= thr
    union = np.logical_or(p, g).sum()
    if union == 0:
        return float("nan")  # 둘 다 완전히 마름 -> 정의되지 않음(평균에서 제외)
    return float(np.logical_and(p, g).sum() / union)


def peak_error_m(pred, gt, dx):
    pi = np.unravel_index(int(pred.argmax()), pred.shape)
    gi = np.unravel_index(int(gt.argmax()), gt.shape)
    return float(np.hypot(pi[0] - gi[0], pi[1] - gi[1]) * dx)


def rain_bin(rain_mm):
    if rain_mm is None:
        return "unknown"
    for lo in (20, 35, 50, 65):
        if rain_mm < lo + 15:
            return f"{lo}-{lo+15}mm"
    return "80mm+"


# ---------------------------------------------------------------- 메인 평가

def evaluate(args):
    with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    contours_dir = os.path.join(ROOT, cfg["contours_dir"])
    flow_dir = os.path.join(ROOT, cfg["flow_dir"])
    image_size = args.image_size or cfg["image_size"]

    train_names, val_names = split_train_val(contours_dir, flow_dir,
                                             val_split=cfg["val_split"], seed=cfg["seed"])
    # --split train: 학습에 쓴 지형에서 같은 지표를 잰다. val 과 비교하면
    # "데이터가 모자라 일반화를 못 하는 것"인지 "목적함수 때문에 학습조차 안 되는 것"인지
    # 갈린다 (train 에서도 피크가 무너지면 데이터 문제가 아니다).
    pool_names = val_names if args.split == "val" else train_names
    print(f"{args.split} 이미지 {len(pool_names)}장 "
          f"(지형 {len({_group_key(n) for n in pool_names})}종)")

    # D4 증강 8장은 같은 지형의 회전본이라 지표가 거의 중복된다. 기본은 r0만 쓴다
    # (등변성은 --equivariance 로 따로 측정).
    names = [n for n in pool_names if f"_{args.aug}_t" in n]
    names.sort()
    if args.limit_terrains:
        # 알파벳 순으로 앞에서 자르면 지역이 한쪽으로 쏠린다(지형 키가 region_ 으로 시작).
        # train/val 을 비교할 때 지역 구성이 달라지면 격차가 교란되므로 seed 로 고정한
        # 무작위 추출을 쓴다.
        keys = sorted({_group_key(n) for n in names})
        rng = np.random.default_rng(cfg["seed"])
        keep = {keys[i] for i in rng.permutation(len(keys))[:args.limit_terrains]}
        names = [n for n in names if _group_key(n) in keep]
    print(f"평가 대상 {len(names)}장 (aug={args.aug}, 지형 {len({_group_key(n) for n in names})}종)")

    rain_values = json.load(open(os.path.join(ROOT, cfg["rain_values_json"]), encoding="utf-8"))
    time_values = json.load(open(os.path.join(ROOT, cfg["time_values_json"]), encoding="utf-8"))

    pred = Predictor(os.path.join(ROOT, args.checkpoint), image_size)
    print(f"체크포인트: {args.checkpoint} (in_ch={pred.in_channels}, cond_dim={pred.cond_dim}, "
          f"device={pred.device})")
    if pred.cond_dim != 2:
        print("  주의: cond_dim != 2 인 구버전 모델입니다. 절대 수심 비교는 의미가 없습니다"
              " (샘플별 자체 정규화로 학습됨). 타깃공간 L1만 참고하세요.")

    gt_src = GroundTruth()
    rows = []
    skipped = 0

    for start in range(0, len(names), args.batch_size):
        chunk = names[start:start + args.batch_size]
        metas, paths, conds = [], [], []
        for n in chunk:
            m = NAME_RE.match(os.path.splitext(n)[0])
            if not m:
                skipped += 1
                continue
            region, sample_name, aug, t_idx = m.group(1), m.group(2), m.group(3), int(m.group(4))
            g = gt_src.get(region, sample_name, aug, t_idx)
            if g is None:
                skipped += 1
                continue
            metas.append((n, region, sample_name, aug, t_idx, g))
            paths.append(os.path.join(contours_dir, n))
            conds.append(pred.cond(rain_values.get(n), time_values.get(n)))
        if not metas:
            continue

        out_t = pred.forward(paths, conds)

        # 타깃공간 L1: 학습 로그의 val L1과 같은 정의 ([-1,1] 공간 평균 절대오차)
        tgt = torch.stack([pred._to_input(os.path.join(flow_dir, m[0])) for m in metas]).to(pred.device)
        l1_each = (out_t - tgt).abs().mean(dim=(1, 2, 3)).cpu().numpy()

        gt_shape = metas[0][5][0].shape
        depths = pred.to_depth(out_t, gt_shape)

        for (n, region, sample_name, aug, t_idx, g), dpred_img, l1 in zip(metas, depths, l1_each):
            dpred = to_array_orientation(dpred_img)
            dgt_raw, dx, rain_mm, t_val = g
            dgt = np.clip(dgt_raw, 0.0, DEPTH_VMAX_M)  # 모델이 표현 가능한 범위로 제한
            if dpred.shape != dgt.shape:
                skipped += 1
                continue
            err = dpred - dgt
            row = {
                "name": n, "region": region, "sample": sample_name, "t_idx": t_idx,
                "rain_mm": rain_mm if rain_mm is not None else "",
                "time_s": round(t_val, 2),
                "mae_m": float(np.abs(err).mean()),
                "rmse_m": float(np.sqrt((err ** 2).mean())),
                "bias_m": float(err.mean()),
                "r": pearson(dpred, dgt),
                "peak_err_m": peak_error_m(dpred, dgt, dx),
                "gt_mean_m": float(dgt.mean()),
                "gt_max_m": float(dgt.max()),
                "pred_mean_m": float(dpred.mean()),
                "pred_max_m": float(dpred.max()),
                "target_l1": float(l1),
                "clipped": float((dgt_raw > DEPTH_VMAX_M).mean()),
            }
            for thr in WET_THRESHOLDS:
                row[f"iou_{int(thr*100)}cm"] = wet_iou(dpred, dgt, thr)
            rows.append(row)

        done = start + len(chunk)
        if done % (args.batch_size * 10) == 0 or done >= len(names):
            print(f"  {done}/{len(names)}", flush=True)

    if skipped:
        print(f"건너뜀 {skipped}장 (원본 OpenFOAM 샘플 없음 또는 파일명 규칙 불일치)")
    return cfg, rows, pred, gt_src, rain_values, time_values, pool_names, contours_dir


# ---------------------------------------------------------------- 강우 단조성

def rain_monotonicity(rows):
    """같은 지형·같은 시점에서 강우량이 더 큰 쪽을 더 깊게 예측하는 비율.

    _rv1/_rv2는 같은 지형에 강우량만 바꿔 다시 돌린 시뮬레이션이므로,
    (지형, t_idx)를 고정하고 강우량이 다른 쌍을 전부 비교한다.
    """
    groups = defaultdict(list)
    for r in rows:
        if r["rain_mm"] == "":
            continue
        base = re.sub(r"_rv\d+$", "", r["sample"])
        groups[(r["region"], base, r["t_idx"])].append(r)

    ok = tie = total = 0
    gt_ok = 0
    for _, items in groups.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if float(a["rain_mm"]) == float(b["rain_mm"]):
                    continue
                lo, hi = (a, b) if float(a["rain_mm"]) < float(b["rain_mm"]) else (b, a)
                total += 1
                if hi["pred_mean_m"] > lo["pred_mean_m"]:
                    ok += 1
                elif hi["pred_mean_m"] == lo["pred_mean_m"]:
                    tie += 1
                if hi["gt_mean_m"] > lo["gt_mean_m"]:
                    gt_ok += 1
    return {"pairs": total, "correct": ok, "tie": tie, "gt_correct": gt_ok}


# ---------------------------------------------------------------- D4 등변성

def equivariance(pred, gt_src, contours_dir, rain_values, time_values, val_names, n_terrains):
    """같은 지형의 8개 회전/반전본 예측을 원래 방향으로 되돌려 서로 비교한다.

    배수는 방향 대칭이므로 정답은 완벽히 등변이다. 예측이 크게 흔들린다면 모델이
    지형이 아니라 등고선 렌더링의 방향성 인공물에 반응하고 있다는 뜻이다.
    """
    inv = {"r0": lambda a: a, "r1": lambda a: np.rot90(a, -1), "r2": lambda a: np.rot90(a, -2),
           "r3": lambda a: np.rot90(a, -3),
           "f0": lambda a: np.fliplr(a), "f1": lambda a: np.fliplr(np.rot90(a, -1)),
           "f2": lambda a: np.fliplr(np.rot90(a, -2)), "f3": lambda a: np.fliplr(np.rot90(a, -3))}

    by_group = defaultdict(list)
    for n in val_names:
        m = NAME_RE.match(os.path.splitext(n)[0])
        if m:
            by_group[(m.group(1), m.group(2), int(m.group(4)))].append((m.group(3), n))

    keys = [k for k in sorted(by_group) if len(by_group[k]) == 8][:n_terrains]
    spreads = []
    for region, sample_name, t_idx in keys:
        items = sorted(by_group[(region, sample_name, t_idx)])
        g = gt_src.get(region, sample_name, "r0", t_idx)
        if g is None:
            continue
        shape = g[0].shape
        paths = [os.path.join(contours_dir, n) for _, n in items]
        conds = [pred.cond(rain_values.get(n), time_values.get(n)) for _, n in items]
        depths = pred.to_depth(pred.forward(paths, conds), shape)
        stack = np.stack([np.ascontiguousarray(inv[aug](to_array_orientation(d)))
                          for (aug, _), d in zip(items, depths)])
        mean = stack.mean(0)
        spreads.append({
            "std_m": float(stack.std(0).mean()),
            "max_dev_m": float(np.abs(stack - mean).max()),
            "mean_depth_m": float(mean.mean()),
        })
    return spreads


# ---------------------------------------------------------------- 리포트

def agg(rows, keys):
    """(그룹 -> 지표 평균) 표. nan은 평균에서 제외한다."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[keys(r)].append(r)
    metrics = ["mae_m", "rmse_m", "r", "peak_err_m", "target_l1", "gt_mean_m"] + \
              [f"iou_{int(t*100)}cm" for t in WET_THRESHOLDS]
    out = {}
    for k, items in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        row = {"n": len(items)}
        for m in metrics:
            vals = [x[m] for x in items if isinstance(x[m], float) and not np.isnan(x[m])]
            row[m] = float(np.mean(vals)) if vals else float("nan")
        out[k] = row
    return out


def md_table(title, table):
    cols = ["n", "mae_m", "rmse_m", "r", "iou_1cm", "iou_5cm", "peak_err_m", "target_l1", "gt_mean_m"]
    head = {"n": "N", "mae_m": "MAE[m]", "rmse_m": "RMSE[m]", "r": "r",
            "iou_1cm": "IoU@1cm", "iou_5cm": "IoU@5cm", "peak_err_m": "최심점오차[m]",
            "target_l1": "타깃L1", "gt_mean_m": "GT평균수심[m]"}
    lines = [f"**{title}**", "",
             "| | " + " | ".join(head[c] for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for k, row in table.items():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(str(v) if c == "n" else ("—" if np.isnan(v) else f"{v:.4f}"))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def save_worst(rows, pred, gt_src, contours_dir, out_dir, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worst = sorted(rows, key=lambda r: -r["mae_m"])[:k]
    fig, axes = plt.subplots(len(worst), 3, figsize=(9, 3 * len(worst)))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(worst):
        m = NAME_RE.match(os.path.splitext(r["name"])[0])
        region, sample_name, aug, t_idx = m.group(1), m.group(2), m.group(3), int(m.group(4))
        g = gt_src.get(region, sample_name, aug, t_idx)
        dgt = np.clip(g[0], 0, DEPTH_VMAX_M)
        dpred = to_array_orientation(pred.to_depth(pred.forward(
            [os.path.join(contours_dir, r["name"])],
            [pred.cond(r["rain_mm"] or None, r["time_s"])]), dgt.shape)[0])
        vmax = max(float(dgt.max()), float(dpred.max()), 1e-3)
        axes[i][0].imshow(Image.open(os.path.join(contours_dir, r["name"])))
        axes[i][0].set_title(f"{r['name']}\nrain={r['rain_mm']}mm t={r['time_s']}s", fontsize=7)
        axes[i][1].imshow(dgt, cmap="Blues", vmin=0, vmax=vmax, origin="lower")
        axes[i][1].set_title(f"GT (max {dgt.max():.3f}m)", fontsize=8)
        axes[i][2].imshow(dpred, cmap="Blues", vmin=0, vmax=vmax, origin="lower")
        axes[i][2].set_title(f"Pred (MAE {r['mae_m']:.4f}m)", fontsize=8)
        for ax in axes[i]:
            ax.set_axis_off()
    fig.tight_layout()
    path = os.path.join(out_dir, f"worst_{k}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--tag", default=None, help="출력 폴더 이름 (기본: 체크포인트 폴더명)")
    p.add_argument("--aug", default="r0", choices=list(D4.keys()))
    p.add_argument("--split", default="val", choices=["val", "train"],
                   help="train 으로 주면 학습에 쓴 지형에서 평가 (과적합/과소적합 판별용)")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit-terrains", type=int, default=0, help="빠른 점검용 지형 수 제한")
    p.add_argument("--equivariance", type=int, default=15, help="D4 등변성 측정 지형 수 (0=생략)")
    p.add_argument("--worst", type=int, default=6, help="최악 사례 그림 장수 (0=생략)")
    args = p.parse_args()

    cfg, rows, pred, gt_src, rain_values, time_values, pool_names, contours_dir = evaluate(args)
    if not rows:
        print("평가된 샘플이 없습니다.")
        return

    tag = args.tag or os.path.basename(os.path.dirname(args.checkpoint))
    out_dir = os.path.join(ROOT, f"eval_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "per_sample.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    overall = agg(rows, lambda r: "전체")
    mono = rain_monotonicity(rows)
    eq = equivariance(pred, gt_src, contours_dir, rain_values, time_values,
                      pool_names, args.equivariance) if args.equivariance else []

    lines = [f"# 평가 리포트 — {tag}", "",
             f"- 체크포인트: `{args.checkpoint}` (in_ch={pred.in_channels}, cond_dim={pred.cond_dim})",
             f"- 설정: `{args.config}`, image_size={args.image_size or cfg['image_size']}, aug={args.aug}",
             f"- 평가 샘플: {len(rows)}장 / 지형 {len({r['sample'] for r in rows})}종 "
             f"(**{args.split}** split)",
             f"- 정답: OpenFOAM 원본 depth [m], {DEPTH_VMAX_M}m로 클리핑",
             "", md_table("전체", overall),
             md_table("지역별", agg(rows, lambda r: r["region"])),
             md_table("강우량 구간별", agg(rows, lambda r: rain_bin(
                 float(r["rain_mm"]) if r["rain_mm"] != "" else None))),
             md_table("시점별", agg(rows, lambda r: f"t{r['t_idx']}")),
             "**강우 단조성** (같은 지형·같은 시점, 강우량만 다른 쌍)", ""]
    if mono["pairs"]:
        lines += [f"- 비교 쌍 {mono['pairs']}개 중 **{mono['correct']}개({mono['correct']/mono['pairs']*100:.1f}%)** 에서 "
                  f"강우가 많은 쪽을 더 깊게 예측 (동점 {mono['tie']})",
                  f"- 참고: 정답(GT)에서도 같은 관계가 성립하는 비율 "
                  f"{mono['gt_correct']}/{mono['pairs']} ({mono['gt_correct']/mono['pairs']*100:.1f}%) "
                  f"— 이 값이 모델이 도달 가능한 상한이다", ""]
    else:
        lines += ["- 비교 가능한 쌍 없음", ""]

    if eq:
        std = float(np.mean([e["std_m"] for e in eq]))
        mdev = float(np.mean([e["max_dev_m"] for e in eq]))
        mdepth = float(np.mean([e["mean_depth_m"] for e in eq]))
        lines += ["**D4 등변성** (같은 지형 8방향 예측을 되돌려 비교, 정답은 완벽히 등변)", "",
                  f"- 지형 {len(eq)}종: 픽셀별 표준편차 평균 **{std:.4f} m** "
                  f"(평균 수심 {mdepth:.4f} m 대비 {std/max(mdepth,1e-9)*100:.1f}%)",
                  f"- 최대 편차 평균 {mdev:.4f} m", ""]

    if args.worst:
        path = save_worst(rows, pred, gt_src, contours_dir, out_dir, args.worst)
        lines += [f"**최악 사례**: `{os.path.basename(path)}`", ""]

    lines += ["---", "", "재현:", "", "```powershell",
              f".venv\\Scripts\\python.exe scripts\\evaluate.py --checkpoint {args.checkpoint} "
              f"--config {args.config} --tag {tag}", "```", ""]

    summary = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary)
    print("\n" + summary)
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
