"""
B안 학습 스크립트: 스칼라 고도격자 -> 스칼라 수심격자 (1ch -> 1ch, FiLM 강우조건).

    python -m src.train_scalar --config configs/config_scalar.yaml

손실 = 젖은픽셀 가중 L1  ( + grad_weight * Sobel gradient L1 )  ( + gan_weight * LSGAN adversarial )

config 플래그로 레버를 켠다:
  wet_weight   수심 큰 픽셀 오차 가중치            (기본 4)
  grad_weight  경계 선명도용 Sobel gradient 손실   (0 = 끔)
  gan_weight   PatchGAN adversarial 손실           (0 = 끔). 켜면 판별자도 같이 학습
  ndf          판별자 폭                            (기본 32)

체크포인트(best.pt / last.pt)에 image_size·ngf·정규화상수가 같이 저장되어
src/inference_scalar.py 가 코드 수정 없이 로드한다.
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.scalar_flow_dataset import ScalarFlowDataset, y_to_depth
from src.models.discriminator import NLayerDiscriminator
from src.models.unet_generator import UnetGenerator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weighted_l1(pred, target, w_pos: float):
    """수심이 큰 곳(=target 이 -1 보다 큰 곳) 오차에 가중치."""
    w = 1.0 + w_pos * (target + 1.0) * 0.5   # dry:1  ~  max-wet:1+w_pos
    return (w * (pred - target).abs()).mean()


_SOBEL_X = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]]) / 8.0
_SOBEL_Y = _SOBEL_X.transpose(-1, -2)


def grad_l1(pred, target):
    """Sobel gradient 의 L1 차이 -> 경계/채널 구조 선명도 유도."""
    kx = _SOBEL_X.to(pred.device)
    ky = _SOBEL_Y.to(pred.device)
    pgx, pgy = F.conv2d(pred, kx, padding=1), F.conv2d(pred, ky, padding=1)
    tgx, tgy = F.conv2d(target, kx, padding=1), F.conv2d(target, ky, padding=1)
    return (pgx - tgx).abs().mean() + (pgy - tgy).abs().mean()


@torch.no_grad()
def evaluate(net, loader, device, stats):
    net.eval()
    l1_norm, l1_mm, n = 0.0, 0.0, 0
    ref, log_max = stats["depth_ref_m"], stats["depth_log_max"]
    for b in loader:
        x, y, c = b["input"].to(device), b["target"].to(device), b["cond"].to(device)
        p = net(x, c)
        l1_norm += (p - y).abs().mean().item()
        d_pred = y_to_depth(p.cpu().numpy(), ref, log_max) * 1000.0
        d_gt = y_to_depth(y.cpu().numpy(), ref, log_max) * 1000.0
        l1_mm += np.abs(d_pred - d_gt).mean()
        n += 1
    net.train()
    return l1_norm / max(n, 1), l1_mm / max(n, 1)


def save_sample(net, loader, out_dir, epoch, device):
    from PIL import Image

    net.eval()
    with torch.no_grad():
        b = next(iter(loader))
        x, y, c = b["input"][:1].to(device), b["target"][:1].to(device), b["cond"][:1].to(device)
        p = net(x, c)

    def blues(arr01):
        a = np.clip(arr01, 0, 1)
        rgb = np.stack([1 - 0.9 * a, 1 - 0.6 * a, np.ones_like(a)], axis=-1)
        return (rgb * 255).astype(np.uint8)

    terr = ((x[0, 0].cpu().numpy() + 1) / 2)
    pred = ((p[0, 0].cpu().numpy() + 1) / 2)
    gt = ((y[0, 0].cpu().numpy() + 1) / 2)
    terr_rgb = (np.stack([terr] * 3, axis=-1) * 255).astype(np.uint8)
    row = np.concatenate([terr_rgb, blues(pred), blues(gt)], axis=1)
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(row).save(os.path.join(out_dir, f"sample_epoch_{epoch:04d}.png"))
    net.train()


def main(cfg: dict) -> None:
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    root = cfg["data_root"]
    img = cfg.get("image_size", 128)
    ngf = cfg.get("ngf", 32)
    ndf = cfg.get("ndf", 32)
    w_pos = cfg.get("wet_weight", 4.0)
    grad_w = cfg.get("grad_weight", 0.0)
    gan_w = cfg.get("gan_weight", 0.0)
    use_flowacc = bool(cfg.get("use_flowacc", False))

    train_set = ScalarFlowDataset(root, "train", image_size=img, augment=True, use_flowacc=use_flowacc)
    val_set = ScalarFlowDataset(root, "val", image_size=img, augment=False, use_flowacc=use_flowacc)
    stats = train_set.stats
    in_ch = train_set.in_channels
    print(f"train {len(train_set)} / val {len(val_set)}  image_size={img} ngf={ngf} in_channels={in_ch} "
          f"flowacc={use_flowacc} wet_weight={w_pos} grad_weight={grad_w} gan_weight={gan_w}")

    train_loader = DataLoader(train_set, batch_size=cfg.get("batch_size", 16), shuffle=True,
                              num_workers=cfg.get("num_workers", 2), drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.get("batch_size", 16), shuffle=False,
                            num_workers=cfg.get("num_workers", 2))

    net = UnetGenerator(in_channels=in_ch, out_channels=1, image_size=img, ngf=ngf,
                        use_film=True, cond_dim=1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.get("lr", 2e-4), betas=(cfg.get("beta1", 0.5), 0.999))
    print(f"netG: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params")

    netD = optD = None
    if gan_w > 0:
        netD = NLayerDiscriminator(in_channels=in_ch + 1, ndf=ndf).to(device)   # (input채널, depth)
        optD = torch.optim.Adam(netD.parameters(), lr=cfg.get("lr", 2e-4), betas=(cfg.get("beta1", 0.5), 0.999))
        print(f"netD: {sum(p.numel() for p in netD.parameters()) / 1e6:.2f}M params (LSGAN)")

    ckpt_dir, out_dir = cfg["checkpoint_dir"], cfg["output_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path, best_path = os.path.join(ckpt_dir, "last.pt"), os.path.join(ckpt_dir, "best.pt")

    start_epoch, best_val, no_improve = 1, float("inf"), 0
    if os.path.exists(last_path):
        st = torch.load(last_path, map_location=device)
        net.load_state_dict(st["netG"])
        if "optimizer_G" in st:
            opt.load_state_dict(st["optimizer_G"])
        if netD is not None and "netD" in st:
            netD.load_state_dict(st["netD"])
            if "optimizer_D" in st:
                optD.load_state_dict(st["optimizer_D"])
        start_epoch = st.get("epoch", 0) + 1
        best_val = st.get("best_val", float("inf"))
        no_improve = st.get("no_improve", 0)
        print(f"재개: epoch {start_epoch} (best_val={best_val:.4f})")

    def save(path, epoch):
        blob = {
            "netG": net.state_dict(), "optimizer_G": opt.state_dict(),
            "epoch": epoch, "best_val": best_val, "no_improve": no_improve,
            "image_size": img, "ngf": ngf, "in_channels": in_ch, "out_channels": 1,
            "use_film": True, "cond_dim": 1, "use_flowacc": use_flowacc,
            "wet_weight": w_pos, "grad_weight": grad_w, "gan_weight": gan_w,
            "stats": {k: stats[k] for k in (
                "depth_ref_m", "depth_max_m", "depth_log_max", "rain_min_mm", "rain_max_mm", "grid_shape")},
        }
        if netD is not None:
            blob["netD"] = netD.state_dict()
            blob["optimizer_D"] = optD.state_dict()
        torch.save(blob, path)

    epochs = cfg.get("epochs", 400)
    patience = cfg.get("patience", 50)
    freq = cfg.get("sample_epoch_freq", 5)
    mse = torch.nn.MSELoss()

    for epoch in range(start_epoch, epochs + 1):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for b in pbar:
            x, y, c = b["input"].to(device), b["target"].to(device), b["cond"].to(device)
            fake = net(x, c)

            if netD is not None:
                # --- D ---
                optD.zero_grad()
                d_real = netD(torch.cat([x, y], 1))
                d_fake = netD(torch.cat([x, fake.detach()], 1))
                loss_d = 0.5 * (mse(d_real, torch.ones_like(d_real)) +
                                mse(d_fake, torch.zeros_like(d_fake)))
                loss_d.backward()
                optD.step()

            # --- G ---
            opt.zero_grad()
            loss_g = weighted_l1(fake, y, w_pos)
            if grad_w > 0:
                loss_g = loss_g + grad_w * grad_l1(fake, y)
            if netD is not None:
                d_fake_g = netD(torch.cat([x, fake], 1))
                loss_g = loss_g + gan_w * mse(d_fake_g, torch.ones_like(d_fake_g))
            loss_g.backward()
            opt.step()
            pbar.set_postfix(g=f"{loss_g.item():.3f}", d=(f"{loss_d.item():.3f}" if netD is not None else "-"))

        if epoch % freq == 0:
            v_norm, v_mm = evaluate(net, val_loader, device, stats)
            print(f"[epoch {epoch}] val L1(norm)={v_norm:.4f}  val L1(depth)={v_mm:.3f} mm")
            save_sample(net, val_loader, out_dir, epoch, device)
            if v_norm < best_val:
                best_val, no_improve = v_norm, 0
                save(best_path, epoch)
                print(f"  -> best 갱신 {best_val:.4f}")
            else:
                no_improve += freq

        save(last_path, epoch)
        if epoch % cfg.get("save_epoch_freq", 25) == 0:
            save(os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"), epoch)
        if no_improve >= patience:
            print(f"조기 종료: 최근 {patience} epoch 개선 없음 (best={best_val:.4f})")
            break

    print("학습 완료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_scalar.yaml")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        main(yaml.safe_load(f))
