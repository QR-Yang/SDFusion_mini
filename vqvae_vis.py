#!/usr/bin/env python3
"""

在 SDFusion_mini 项目根目录下运行：
    python vqvae_vis.py \
        --ckpt checkpoints/vqvae_1e-4.pth \
        --n 8 \
        --out results/vis

输出：
    results/vis/<synset>_<model_id>_gt.obj   ← 原始 GT mesh
    results/vis/<synset>_<model_id>_rec.obj  ← encode→decode 重建 mesh
"""

import argparse, glob, os, sys
import numpy as np
import torch
import h5py

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="checkpoints/vqvae_1e-4.pth", help="checkpoint 路径")
p.add_argument("--n",    type=int, default=8,          help="测几个样本")
p.add_argument("--out",  default="results/vis",        help="输出目录")
p.add_argument("--gpu",  type=int, default=0)
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"[*] device: {device}")

size_mb = os.path.getsize(args.ckpt) / 1024 / 1024


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vqvae import VQVAE
from config import vqvae_config

model = VQVAE(
    ddconfig=vqvae_config["ddconfig"],   # ch_mult=[1,2,4], num_res_blocks=1
    n_embed=vqvae_config["n_embed"],     # 8192
    embed_dim=vqvae_config["embed_dim"], # 3
).to(device)

print(f"[*] 加载 checkpoint: {args.ckpt}")
state = torch.load(args.ckpt, map_location=device)
model.load_state_dict(state)
model.eval()
print("[*] 模型加载完成")

DATA_ROOT = "data/ShapeNet/SDF_v1/resolution_64"
SYNSETS   = ["03001627"]

h5_files = []
for syn in SYNSETS:
    found = sorted(glob.glob(os.path.join(DATA_ROOT, syn, "*", "ori_sample_grid.h5")))
    h5_files.extend(found)
    if len(h5_files) >= args.n:
        break

h5_files = h5_files[:args.n]

print(f"[*] 共找到 {len(h5_files)} 个样本，开始 encode→decode")


try:
    import mcubes
    def save_obj(sdf_np, path):
        verts, faces = mcubes.marching_cubes(-sdf_np, 0)
        if len(verts) == 0:
            print(f"    [!] 空 mesh（SDF 全正），跳过: {path}")
            return
        verts = (verts / (sdf_np.shape[0] - 1.0)) * 2.0 - 1.0
        mcubes.export_obj(verts, faces, path)
    HAS_MCUBES = True
except ImportError:
    HAS_MCUBES = False
    print("[!] 未安装 PyMCubes，只输出 .npy（pip install PyMCubes 可导出 .obj）")


for h5_path in h5_files:
    parts    = h5_path.replace("\\", "/").split("/")
    synset   = parts[-3]
    model_id = parts[-2]
    name     = f"{synset}_{model_id}"
    print(f"  {name} ...", end=" ", flush=True)

    with h5py.File(h5_path, "r") as f:
        sdf_np = f["pc_sdf_sample"][:].astype(np.float32).reshape(64, 64, 64)
    sdf_np = np.clip(sdf_np, -0.2, 0.2)   # 与训练时一致

    x = torch.from_numpy(sdf_np).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,64,64,64)

    with torch.no_grad():
        quant, _, _ = model.encode(x)
        rec = model.decode(quant)

    rec_np = rec.squeeze().cpu().numpy()   # (64,64,64)

    np.save(os.path.join(args.out, f"{name}_gt.npy"),  sdf_np)
    np.save(os.path.join(args.out, f"{name}_rec.npy"), rec_np)

    if HAS_MCUBES:
        save_obj(sdf_np, os.path.join(args.out, f"{name}_gt.obj"))
        save_obj(rec_np, os.path.join(args.out, f"{name}_rec.obj"))
        print("→ _gt.obj + _rec.obj ✓")
    else:
        print("→ _gt.npy + _rec.npy ✓")

print(f"\n[*] 结果保存到 {args.out}/")
