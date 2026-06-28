import os
os.environ["TORCH_BLAS_PREFER_CUBLASLT"] = "0"
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"

import random
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from tqdm import tqdm

from chair_config import vqvae_chair_config
from text_dataset import get_text_dataloader
from text_encoder import BERTTextEncoder
from diffusion_unet.text_model import TextConditionedUNet3D
from vqvae import VQVAE

# ── 超参数 ──────────────────────────────────────────────────
batch_size = 8
num_epochs = 200
lr = 1e-5
num_train_timesteps = 1000
p_uncond = 0.1                     # classifier-free guidance：训练时丢弃文本的概率

vqvae_path = "checkpoints/vqvae_chair_best.pth"
save_path = "checkpoints/txt2shape_chair.pth"


def atomic_save(state, path):
    """先写临时文件再原子替换，避免训练中断时把 checkpoint 写坏。"""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def train():
    device = "cuda"

    # 数据：chair 的 (SDF, caption) 配对
    dataloader = get_text_dataloader(batch_size=batch_size, phase="train")

    # 冻结的 VQVAE（只做 encode）
    vqvae = VQVAE(
        ddconfig=vqvae_chair_config["ddconfig"],
        n_embed=vqvae_chair_config["n_embed"],
        embed_dim=vqvae_chair_config["embed_dim"],
    ).to(device)
    vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    # 冻结的 BERT 文本编码器
    text_encoder = BERTTextEncoder(
        model_name="/data1/xuc/SDFusion_mini/pretrained/bert-base-uncased",
        device=device,
    )

    # 可训练的文本条件 UNet
    model = TextConditionedUNet3D(context_dim=text_encoder.output_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for sdf, captions in tqdm(dataloader):
            sdf = sdf.to(device)
            captions = list(captions)

            # classifier-free guidance：以一定概率把整条文本替换成空串
            captions = ["" if random.random() < p_uncond else c for c in captions]

            with torch.no_grad():
                # 连续 latent (B, 3, 16, 16, 16)
                z = vqvae(sdf, forward_no_quant=True, encode_only=True)
                context = text_encoder(captions)          # (B, L, 768)

            noise = torch.randn_like(z)
            t = torch.randint(0, num_train_timesteps, (z.shape[0],), device=device)
            zt = scheduler.add_noise(z, noise, t)

            noise_pred = model(zt, t, context)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print("epoch:", epoch + 1, "loss:", total_loss / len(dataloader))
        os.makedirs("checkpoints", exist_ok=True)
        atomic_save(model.state_dict(), save_path)


if __name__ == "__main__":
    train()
