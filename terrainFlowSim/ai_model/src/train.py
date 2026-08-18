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

from src.datasets.contour_flow_dataset import ContourFlowDataset, split_train_val
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

    train_names, val_names = split_train_val(
        cfg["contours_dir"], cfg["flow_dir"], val_split=cfg["val_split"], seed=cfg["seed"]
    )
    train_set = ContourFlowDataset(cfg["contours_dir"], cfg["flow_dir"],
                                    image_size=cfg["image_size"], augment=True, names=train_names)
    val_set = ContourFlowDataset(cfg["contours_dir"], cfg["flow_dir"],
                                  image_size=cfg["image_size"], augment=False, names=val_names) \
        if val_names else None

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"], drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"]) if val_set else None

    print(f"train samples: {len(train_set)}, val samples: {len(val_set) if val_set else 0}")

    model = TranslationModel(
        mode=cfg["mode"], in_channels=3, out_channels=3, image_size=cfg["image_size"],
        lr=cfg["lr"], beta1=cfg["beta1"], lambda_l1=cfg["lambda_l1"], gan_loss=cfg["gan_loss"],
        device=device,
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

    global_step = 0
    for epoch in range(start_epoch, cfg["epochs"] + 1):
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
    save_triplet_png(model.real_input[0], model.fake_target[0], model.real_target[0], path)
    model.netG.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
