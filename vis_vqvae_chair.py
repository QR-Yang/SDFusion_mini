import os
os.environ["TORCH_BLAS_PREFER_CUBLASLT"] = "0"
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"

import argparse
import glob
import numpy as np
import torch
import h5py

from chair_config import vqvae_chair_config
from vqvae import VQVAE

SYNSET = "03001627"
SDF_ROOT = "data/ShapeNet/SDF_v1/resolution_64"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/vqvae_chair_best.pth")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", default="results/vqvae_chair")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[*] device: {device}")

    model = VQVAE(
        ddconfig=vqvae_chair_config["ddconfig"],
        n_embed=vqvae_chair_config["n_embed"],
        embed_dim=vqvae_chair_config["embed_dim"],
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"[*] loaded {args.ckpt}")

    h5_files = sorted(glob.glob(
        os.path.join(SDF_ROOT, SYNSET, "*", "ori_sample_grid.h5")))[:args.n]
    print(f"[*] {len(h5_files)} 个样本")

    try:
        import mcubes
        def save_obj(sdf_np, path):
            verts, faces = mcubes.marching_cubes(-sdf_np, 0.00)
            if len(verts) == 0:
                print(f"    [!] 空 mesh: {path}")
                return
            verts = (verts / (sdf_np.shape[0] - 1.0)) * 2.0 - 1.0
            mcubes.export_obj(verts, faces, path)
        HAS_MCUBES = True
    except ImportError:
        HAS_MCUBES = False
        print("[!] 未装 PyMCubes，只存 .npy")

    for h5_path in h5_files:
        mid = os.path.basename(os.path.dirname(h5_path))
        with h5py.File(h5_path, "r") as f:
            sdf_np = f["pc_sdf_sample"][:].astype(np.float32).reshape(64, 64, 64)
        sdf_np = np.clip(sdf_np, -0.2, 0.2)

        x = torch.from_numpy(sdf_np).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            quant, _, _ = model.encode(x)
            rec = model.decode(quant)
        rec_np = rec.squeeze().cpu().numpy()

        l1 = np.abs(rec_np - sdf_np).mean()
        print(f"  {mid}: rec_l1={l1:.5f}")

        np.save(os.path.join(args.out, f"{mid}_gt.npy"), sdf_np)
        np.save(os.path.join(args.out, f"{mid}_rec.npy"), rec_np)
        if HAS_MCUBES:
            save_obj(sdf_np, os.path.join(args.out, f"{mid}_gt.obj"))
            save_obj(rec_np, os.path.join(args.out, f"{mid}_rec.obj"))

    print(f"[*] 输出到 {args.out}/")


if __name__ == "__main__":
    main()
