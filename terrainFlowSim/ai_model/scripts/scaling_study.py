"""
데이터 스케일링 곡선 (IMPROVEMENT_GUIDE.md Phase 1).

답하려는 질문: **"지형을 더 확보하면 좋아지는가, 좋아진다면 얼마나 더 필요한가?"**

OpenFOAM 한 샷이 10~20분이라 지형 200개 추가는 며칠짜리 투자다. 그 전에
"지형 수만 바꾼" 학습을 몇 지점 돌려서 곡선의 기울기를 본다.

- 학습 지형 수만 바꾸고 나머지 설정은 전부 고정 (`max_train_groups`)
- **val 셋은 네 지점 모두 동일** — split 을 먼저 하고 train 쪽만 부분집합을 뽑으므로 보장됨
- 절대 성능이 아니라 기울기를 보는 실험이라 축소 설정(128px/ngf32)으로 돈다

판정:
- 마지막 구간에서 아직 뚜렷하게 개선 -> 데이터가 병목. 추가 투자 정당함
- 평평 -> 데이터 추가는 낭비. 다른 레버로
- 지역별로 갈리면(gangwon 만 살아있음) 전체가 아니라 그 지역만 추가

주의: 이 곡선은 **문제 B(공간 패턴 일반화, r·IoU)** 에만 답한다. 문제 A(피크 평탄화)는
학습 지형에서도 똑같이 나타나므로 여기서 답이 나오지 않는다 (Phase 0-B 참조).

사용법:
    .venv\\Scripts\\python.exe scripts\\scaling_study.py --base configs/config_p2_wet8.yaml \
        --points 75 150 300 0        # 0 = 제한 없음(전체)
    .venv\\Scripts\\python.exe scripts\\scaling_study.py --report-only   # 이미 돈 결과만 표로
"""

import os
import re
import sys
import csv
import copy
import json
import argparse
import subprocess

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(PY):
    PY = sys.executable


def config_for(base_cfg, base_path, n_groups):
    """지형 수만 바꾼 config 를 만들어 저장하고 (경로, 태그) 반환."""
    tag = f"scale{n_groups if n_groups else 'full'}"
    cfg = copy.deepcopy(base_cfg)
    if n_groups:
        cfg["max_train_groups"] = int(n_groups)
    else:
        cfg.pop("max_train_groups", None)
    cfg["checkpoint_dir"] = f"checkpoints_{tag}"
    cfg["output_dir"] = f"outputs_{tag}"
    path = os.path.join(ROOT, "configs", f"config_{tag}.yaml")
    header = (f"# 자동 생성 (scripts/scaling_study.py). 기반: {os.path.basename(base_path)}\n"
              f"# 학습 지형 수 = {n_groups if n_groups else '전체'} 외에는 기반 config 와 동일.\n"
              f"# 직접 수정하지 말고 기반 config 를 고친 뒤 다시 생성할 것.\n\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path, tag


def run(cmd, log_path):
    print(f"  $ {' '.join(os.path.basename(c) if os.sep in c else c for c in cmd)}")
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode


def read_summary(tag):
    """evaluate.py 가 남긴 summary.md 의 '전체' 행을 파싱."""
    path = os.path.join(ROOT, f"eval_{tag}", "summary.md")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("| 전체 |"):
                c = [x.strip() for x in line.strip().strip("|").split("|")]
                keys = ["n", "mae_m", "rmse_m", "r", "iou_1cm", "iou_5cm",
                        "peak_err_m", "target_l1", "gt_mean_m"]
                out = {}
                for k, v in zip(keys, c[1:]):
                    try:
                        out[k] = float(v)
                    except ValueError:
                        out[k] = float("nan")
                return out
    return None


def peak_ratio(tag):
    """per_sample.csv 에서 최대수심 비율 중앙값 — 문제 A 의 핵심 지표."""
    path = os.path.join(ROOT, f"eval_{tag}", "per_sample.csv")
    if not os.path.exists(path):
        return float("nan")
    import numpy as np
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    if not rows:
        return float("nan")
    px = np.array([float(r["pred_max_m"]) for r in rows])
    gx = np.array([float(r["gt_max_m"]) for r in rows])
    return float(np.median(px / np.maximum(gx, 1e-9)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="configs/config_p2_wet8.yaml",
                   help="기반 config. Phase 2 에서 이긴 설정을 쓸 것")
    p.add_argument("--points", nargs="+", type=int, default=[75, 150, 300, 0],
                   help="학습에 쓸 지형 수들. 0 = 제한 없음(전체)")
    p.add_argument("--report-only", action="store_true", help="학습/평가 없이 기존 결과만 표로")
    args = p.parse_args()

    base_path = os.path.join(ROOT, args.base)
    with open(base_path, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    results = []
    for n in args.points:
        cfg_path, tag = config_for(base_cfg, base_path, n)
        label = str(n) if n else "전체"

        if not args.report_only:
            ck = os.path.join(ROOT, f"checkpoints_{tag}", "best.pt")
            if os.path.exists(ck):
                print(f"[{label}] 체크포인트 있음 -> 학습 건너뜀")
            else:
                print(f"[{label}] 학습")
                rc = run([PY, "-m", "src.train", "--config", os.path.relpath(cfg_path, ROOT)],
                         os.path.join(ROOT, f"train_{tag}.log"))
                if rc != 0:
                    print(f"  학습 실패 (exit {rc}) -> 건너뜀. train_{tag}.log 확인")
                    continue
            print(f"[{label}] 평가")
            run([PY, os.path.join("scripts", "evaluate.py"),
                 "--checkpoint", f"checkpoints_{tag}/best.pt",
                 "--config", os.path.relpath(cfg_path, ROOT),
                 "--tag", tag, "--equivariance", "0", "--worst", "0"],
                os.path.join(ROOT, f"eval_{tag}.log"))

        s = read_summary(tag)
        if s:
            s["label"] = label
            s["peak_ratio"] = peak_ratio(tag)
            results.append(s)

    if not results:
        print("결과 없음.")
        return

    print("\n## 데이터 스케일링 곡선 (val 셋 동일)\n")
    print("| 학습 지형 수 | MAE[m] | r | IoU@1cm | IoU@5cm | 최심점오차[m] | 최대수심 비율 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for s in results:
        print(f"| {s['label']} | {s['mae_m']:.4f} | {s['r']:.4f} | {s['iou_1cm']:.4f} | "
              f"{s['iou_5cm']:.4f} | {s['peak_err_m']:.1f} | {s['peak_ratio']:.3f} |")

    if len(results) >= 2:
        a, b = results[-2], results[-1]
        d_r, d_iou = b["r"] - a["r"], b["iou_5cm"] - a["iou_5cm"]
        print(f"\n마지막 구간({a['label']} -> {b['label']}): r {d_r:+.4f}, IoU@5cm {d_iou:+.4f}")
        if d_r > 0.02 or d_iou > 0.02:
            print("-> 아직 우하향. 지형 추가가 값어치 있다 (OpenFOAM 투자 정당).")
        else:
            print("-> 평평. 지형을 더 늘려도 이 설정에서는 이득이 작다. 다른 레버로.")
    print("\n주의: 이 곡선은 문제 B(공간 패턴 일반화)에만 답한다. 최대수심 비율이 "
          "지형 수에 따라 거의 안 변한다면 문제 A는 여전히 목적함수 쪽이다.")

    with open(os.path.join(ROOT, "scaling_curve.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("저장: scaling_curve.json")


if __name__ == "__main__":
    main()
