"""
학습 스크립트.

사용법 (ai_model/ 디렉터리에서, 가상환경 활성화 후):
    python -m src.train --config configs/config.yaml

data/contours, data/flow가 비어 있으면 바로 에러를 내며 안내 메시지를 보여준다.
"""

import argparse
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.contour_flow_dataset import ContourFlowDataset, split_train_val, _group_key
from src.models.translation_model import TranslationModel
from src.utils.image_utils import save_triplet_png


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    rain_dir = cfg.get("rain_dir")  # 주면 등고선(3채널)+강우(1채널)=4채널 입력 (채널 결합 방식)
    rain_values_json = cfg.get("rain_values_json")  # 주면 강우량을 FiLM 조건 스칼라로 사용
    time_values_json = cfg.get("time_values_json")  # 주면 시뮬레이션 경과 시점도 FiLM 조건 스칼라로 사용
    use_film = bool(rain_values_json or time_values_json)
    # cond_dim: rain/time 중 실제로 켜진 조건 개수만큼 (둘 다 켜면 2, 하나만 켜면 1)
    cond_dim = int(bool(rain_values_json)) + int(bool(time_values_json))
    if cond_dim == 0:
        cond_dim = 1  # use_film=False라 실제로 안 쓰이지만 생성자 인자 형식상 기본값
    in_channels = cfg.get("in_channels", 4 if rain_dir else 3)

    train_names, val_names = split_train_val(
        cfg["contours_dir"], cfg["flow_dir"], val_split=cfg["val_split"], seed=cfg["seed"]
    )

    # max_train_groups: 학습에 쓸 "고유 지형" 수를 제한한다 (데이터 스케일링 곡선용,
    # IMPROVEMENT_GUIDE.md Phase 1). val 은 건드리지 않으므로 지형 수만 다른 여러 학습을
    # 같은 val 셋으로 비교할 수 있다. 어느 지형이 뽑히는지는 seed 로 고정한다.
    max_groups = cfg.get("max_train_groups")
    if max_groups:
        import numpy as _np
        groups = {}
        for n in train_names:
            groups.setdefault(_group_key(n), []).append(n)
        keys = sorted(groups)
        rng = _np.random.default_rng(cfg["seed"])
        keep = {keys[i] for i in rng.permutation(len(keys))[:int(max_groups)]}
        train_names = [n for n in train_names if _group_key(n) in keep]
        print(f"학습 지형 제한: {len(keys)}종 -> {len(keep)}종 ({len(train_names)}장)")
    train_set = ContourFlowDataset(cfg["contours_dir"], cfg["flow_dir"],
                                    image_size=cfg["image_size"], augment=True, names=train_names,
                                    rain_dir=rain_dir, rain_values_json=rain_values_json,
                                    time_values_json=time_values_json)
    val_set = ContourFlowDataset(cfg["contours_dir"], cfg["flow_dir"],
                                  image_size=cfg["image_size"], augment=False, names=val_names,
                                  rain_dir=rain_dir, rain_values_json=rain_values_json,
                                  time_values_json=time_values_json) \
        if val_names else None

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"], drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"]) if val_set else None

    cond_note = " (강우 조건 포함, rain_dir=" + str(rain_dir) + ")" if rain_dir else \
        (f" (FiLM 조건 {cond_dim}차원: rain={bool(rain_values_json)}, time={bool(time_values_json)})"
         if use_film else "")
    print(f"train samples: {len(train_set)}, val samples: {len(val_set) if val_set else 0}, "
          f"in_channels: {in_channels}{cond_note}")

    model = TranslationModel(
        mode=cfg["mode"], in_channels=in_channels, out_channels=3, image_size=cfg["image_size"],
        lr=cfg["lr"], beta1=cfg["beta1"], lambda_l1=cfg["lambda_l1"], gan_loss=cfg["gan_loss"],
        device=device, use_film=use_film, cond_dim=cond_dim, ngf=cfg.get("ngf", 64),
        wet_weight=cfg.get("wet_weight", 1.0),
    )

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # 중단 후 재개용: 매 epoch 끝에 last.pt를 저장해두고, 시작할 때 있으면 이어서 진행
    last_ckpt_path = os.path.join(cfg["checkpoint_dir"], "last.pt")
    best_ckpt_path = os.path.join(cfg["checkpoint_dir"], "best.pt")

    start_epoch = 1
    best_val_l1 = float("inf")
    epochs_no_improve = 0
    if os.path.exists(last_ckpt_path):
        state = model.load_checkpoint(last_ckpt_path, map_location=device, load_optimizer=True)
        start_epoch = state.get("epoch", 0) + 1
        best_val_l1 = state.get("best_val_l1", float("inf"))
        epochs_no_improve = state.get("epochs_no_improve", 0)
        print(f"체크포인트에서 재개: epoch {start_epoch}부터 (best_val_l1={best_val_l1:.4f}, "
              f"개선 없던 epoch 수={epochs_no_improve})")

    # 조기 종료: sample_epoch_freq마다 검증해서 patience 동안 개선이 없으면 멈춤
    patience = cfg.get("patience", 30)

    # 학습률 선형 감소 (pix2pix 표준 기법): lr_decay_start epoch까지는 고정,
    # 그 이후 epochs까지 선형으로 0에 가깝게 줄인다. GAN 모드에서 판별자가
    # 생성자를 압도하는 문제 완화에 도움이 될 수 있다.
    lr_decay_start = cfg.get("lr_decay_start", None)

    def lr_factor(epoch):
        if not lr_decay_start or epoch <= lr_decay_start:
            return 1.0
        total_decay_epochs = max(1, cfg["epochs"] - lr_decay_start)
        return max(0.0, 1.0 - (epoch - lr_decay_start) / total_decay_epochs)

    def apply_lr(epoch):
        factor = lr_factor(epoch)
        for pg in model.optimizer_G.param_groups:
            pg["lr"] = cfg["lr"] * factor
        if model.optimizer_D is not None:
            for pg in model.optimizer_D.param_groups:
                pg["lr"] = cfg["lr"] * factor
        return factor

    global_step = 0
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        lr_now_factor = apply_lr(epoch)
        if lr_decay_start and epoch % cfg["sample_epoch_freq"] == 0:
            print(f"[epoch {epoch}] lr = {cfg['lr'] * lr_now_factor:.6f}")

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg['epochs']}")
        for batch in pbar:
            model.set_input(batch)
            losses = model.optimize_parameters()
            global_step += 1

            if global_step % cfg["log_iter_freq"] == 0:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})

        if val_loader is not None and epoch % cfg["sample_epoch_freq"] == 0:
            val_l1 = evaluate(model, val_loader)
            print(f"[epoch {epoch}] val L1: {val_l1:.4f}")
            save_sample(model, val_loader, cfg["output_dir"], epoch)

            if val_l1 < best_val_l1:
                best_val_l1 = val_l1
                epochs_no_improve = 0
                model.save_checkpoint(best_ckpt_path, extra={"epoch": epoch, "best_val_l1": best_val_l1})
                print(f"  -> best 갱신 ({best_val_l1:.4f}), 저장: {best_ckpt_path}")
            else:
                epochs_no_improve += cfg["sample_epoch_freq"]

        if epoch % cfg["save_epoch_freq"] == 0 or epoch == cfg["epochs"]:
            ckpt_path = os.path.join(cfg["checkpoint_dir"], f"epoch_{epoch:04d}.pt")
            model.save_checkpoint(ckpt_path, extra={"epoch": epoch, "best_val_l1": best_val_l1})
            print(f"체크포인트 저장: {ckpt_path}")

        # 매 epoch 끝에 재개용 체크포인트 갱신 (중단되어도 여기서부터 다시 시작 가능)
        model.save_checkpoint(last_ckpt_path, extra={
            "epoch": epoch, "best_val_l1": best_val_l1, "epochs_no_improve": epochs_no_improve,
        })

        if epochs_no_improve >= patience:
            print(f"조기 종료: epoch {epoch} 기준 최근 {patience} epoch 동안 val L1 개선 없음 "
                  f"(best={best_val_l1:.4f})")
            break

    print("학습 완료")


@torch.no_grad()
def evaluate(model, val_loader):
    model.netG.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        model.set_input(batch)
        model.forward()
        total += torch.nn.functional.l1_loss(model.fake_target, model.real_target).item()
        n += 1
    model.netG.train()
    return total / max(n, 1)


@torch.no_grad()
def save_sample(model, val_loader, output_dir, epoch):
    model.netG.eval()
    batch = next(iter(val_loader))
    model.set_input(batch)
    model.forward()
    path = os.path.join(output_dir, f"sample_epoch_{epoch:04d}.png")
    # real_input이 4채널(등고선+강우)이어도 미리보기는 등고선(RGB) 부분만 보여준다.
    save_triplet_png(model.real_input[0][:3], model.fake_target[0], model.real_target[0], path)
    model.netG.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
