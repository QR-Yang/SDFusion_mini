import os
os.environ["TORCH_BLAS_PREFER_CUBLASLT"] = "0"
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"

import argparse
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from chair_config import vqvae_chair_config
from vqvae import VQVAE

SYNSET = "03001627"          # chair
SDF_ROOT = "data/ShapeNet/SDF_v1/resolution_64"
LIST_DIR = "dataset_info_files/ShapeNet_filelists"
TRUNC = 0.2                  # SDF 截断阈值，与 datasets.py 一致


# ──────────────────────────────────────────────────────────────
# 单类 chair 数据集
# ──────────────────────────────────────────────────────────────
class ChairSDFDataset(Dataset):
    def __init__(self, phase="train"):
        lst = os.path.join(LIST_DIR, f"{SYNSET}_{phase}.lst")
        assert os.path.isfile(lst), f"找不到列表文件 {lst}"
        self.items = []
        missing = 0
        with open(lst) as f:
            for mid in f:
                mid = mid.strip()
                if not mid:
                    continue
                p = os.path.join(SDF_ROOT, SYNSET, mid, "ori_sample_grid.h5")
                if os.path.isfile(p):
                    self.items.append(p)
                else:
                    missing += 1
        print(f"[ChairSDFDataset/{phase}] {len(self.items)} 个样本"
              f"（缺失 {missing} 个）")
        assert len(self.items) > 0, "没有可用的 chair SDF，请检查数据路径"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        with h5py.File(self.items[i], "r") as f:
            sdf = f["pc_sdf_sample"][:].astype(np.float32)
        sdf = torch.from_numpy(sdf).view(1, 64, 64, 64)
        sdf = torch.clamp(sdf, -TRUNC, TRUNC)
        return sdf


def atomic_save(state, path):
    """先写临时文件再原子替换，避免训练中断时把 checkpoint 写坏。"""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--codebook_weight", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=20,
                   help="每多少个 epoch 额外存一个带 epoch 号的快照")
    args = p.parse_args()

    device = "cuda"
    os.makedirs(args.save_dir, exist_ok=True)
    latest_path = os.path.join(args.save_dir, "vqvae_chair.pth")
    best_path = os.path.join(args.save_dir, "vqvae_chair_best.pth")

    loader = DataLoader(
        ChairSDFDataset("train"),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )

    model = VQVAE(
        ddconfig=vqvae_chair_config["ddconfig"],
        n_embed=vqvae_chair_config["n_embed"],
        embed_dim=vqvae_chair_config["embed_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    print(f"[*] 单类 chair VQVAE 训练（纯 L1，对齐原版逻辑）")
    print(f"[*] epochs={args.epochs} batch={args.batch_size} lr={args.lr}")
    print(f"[*] ddconfig: num_res_blocks="
          f"{vqvae_chair_config['ddconfig']['num_res_blocks']}, "
          f"latent=16^3 x {vqvae_chair_config['embed_dim']}")

    best_rec = float("inf")
    for epoch in range(args.epochs):
        model.train()
        tot_loss, tot_rec, n = 0.0, 0.0, 0
        for sdf in tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            sdf = sdf.to(device)
            recon, codebook_loss = model(sdf)

            # 纯 L1 + codebook commitment，和原版 train_vqvae.py 一致
            rec_loss = F.l1_loss(recon, sdf)
            loss = rec_loss + args.codebook_weight * codebook_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            tot_rec += rec_loss.item()
            n += 1

        scheduler.step()
        mean_loss = tot_loss / n
        mean_rec = tot_rec / n
        print(f"epoch {epoch+1}: loss={mean_loss:.5f} "
              f"rec_l1={mean_rec:.5f} lr={scheduler.get_last_lr()[0]:.2e}")

        # 每个 epoch 安全覆盖 latest
        atomic_save(model.state_dict(), latest_path)
        # 重建最好的存 best
        if mean_rec < best_rec:
            best_rec = mean_rec
            atomic_save(model.state_dict(), best_path)
        # 定期快照（带 epoch 号，便于回滚）
        if (epoch + 1) % args.save_every == 0:
            snap = os.path.join(args.save_dir, f"vqvae_chair_ep{epoch+1}.pth")
            atomic_save(model.state_dict(), snap)

    print(f"[*] 完成。latest={latest_path}  best={best_path} (rec_l1={best_rec:.5f})")


if __name__ == "__main__":
    main()

