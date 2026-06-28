import os
os.environ["TORCH_BLAS_PREFER_CUBLASLT"] = "0"
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"

import argparse
import torch
import trimesh
from diffusers import DDIMScheduler
from skimage import measure

from chair_config import vqvae_chair_config
from text_encoder import BERTTextEncoder
from diffusion_unet import TextConditionedUNet3D
from vqvae import VQVAE

# ── 默认配置 ────────────────────────────────────────────────
num_train_timesteps = 1000
vqvae_path = "checkpoints/vqvae_chair_best.pth"
txt2shape_path = "checkpoints/txt2shape_chair.pth"
save_dir = "outputs"


def save_obj(sdf, path):
    """sdf: (1, 64, 64, 64) numpy；用 marching cubes 导出 obj。"""
    sdf = sdf[0]
    try:
        verts, faces, normals, values = measure.marching_cubes(sdf, level=0.0)
    except (ValueError, RuntimeError):
        print(f"    [!] marching cubes 失败（可能 SDF 全正/全负），跳过 {path}")
        return False
    verts = verts / 63.0 - 0.5
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(path)
    return True


def sample(prompt, num_samples, num_inference_steps, guidance_scale, out_prefix):
    device = "cuda"

    # 冻结 VQVAE（只 decode）
    vqvae = VQVAE(
        ddconfig=vqvae_chair_config["ddconfig"],
        n_embed=vqvae_chair_config["n_embed"],
        embed_dim=vqvae_chair_config["embed_dim"],
    ).to(device)
    vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
    vqvae.eval()

    # BERT 文本编码器
    text_encoder = BERTTextEncoder(device=device)

    # 文本条件 UNet
    model = TextConditionedUNet3D(context_dim=text_encoder.output_dim).to(device)
    model.load_state_dict(torch.load(txt2shape_path, map_location=device))
    model.eval()

    # DDIM 采样器（比全步 DDPM 快很多）
    scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
    scheduler.set_timesteps(num_inference_steps)

    # 文本 context：条件 + 无条件（空串），用于 classifier-free guidance
    cond_context = text_encoder([prompt] * num_samples)        # (N, L, 768)
    uncond_context = text_encoder([""] * num_samples)          # (N, L, 768)

    z = torch.randn(num_samples, 3, 16, 16, 16).to(device)

    with torch.no_grad():
        for t in scheduler.timesteps:
            ts = torch.tensor([t] * num_samples, device=device).long()
            noise_cond = model(z, ts, cond_context)
            if guidance_scale != 1.0:
                noise_uncond = model(z, ts, uncond_context)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = noise_cond
            z = scheduler.step(noise_pred, t, z).prev_sample

        sdf = vqvae.decode_no_quant(z)        # (N, 1, 64, 64, 64)

    os.makedirs(save_dir, exist_ok=True)
    sdf = sdf.cpu().numpy()
    saved = 0
    for i in range(num_samples):
        path = os.path.join(save_dir, f"{out_prefix}_{i}.obj")
        if save_obj(sdf[i], path):
            saved += 1
    print(f"[*] prompt: \"{prompt}\"")
    print(f"[*] 成功保存 {saved}/{num_samples} 个 mesh 到 {save_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="a chair with armrests and four legs",
                   help="文本描述")
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--steps", type=int, default=100, help="DDIM 推理步数")
    p.add_argument("--guidance_scale", type=float, default=3.0,
                   help="classifier-free guidance 强度，1.0 表示不用 CFG")
    p.add_argument("--out_prefix", type=str, default="chair",
                   help="输出 obj 文件名前缀")
    args = p.parse_args()

    sample(
        prompt=args.prompt,
        num_samples=args.num_samples,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        out_prefix=args.out_prefix,
    )
