import os
os.environ["TORCH_BLAS_PREFER_CUBLASLT"] = "0"
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from tqdm import tqdm
from config import vqvae_config
from datasets import get_dataloader
from diffusion_unet import ClassConditionedUNet3D
from vqvae import VQVAE
batch_size = 12
num_epochs = 30
lr = 2e-5
num_train_timesteps = 5000
vqvae_path = "checkpoints/vqvae_1e-4.pth"
save_path = "checkpoints/class_diffusion_2e-5.pth"
def train():
    device = "cuda"
    dataloader = get_dataloader(batch_size=batch_size)
    vqvae = VQVAE(
        ddconfig=vqvae_config["ddconfig"],
        n_embed=vqvae_config["n_embed"],
        embed_dim=vqvae_config["embed_dim"],
    ).to(device)
    vqvae.load_state_dict(torch.load(vqvae_path))
    vqvae.eval()
    model = ClassConditionedUNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    for epoch in range(num_epochs):
        total_loss = 0
        for data in tqdm(dataloader):
            sdf = data[0].to(device)
            class_id = data[1].to(device)
            with torch.no_grad():
                z = vqvae(sdf, forward_no_quant=True, encode_only=True)
            noise = torch.randn_like(z)
            t = torch.randint(0, num_train_timesteps, (z.shape[0],), device=device)
            zt = scheduler.add_noise(z, noise, t)
            noise_pred = model(zt, t, class_id)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("epoch:", epoch + 1, "loss:", total_loss / len(dataloader))
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), save_path)
if __name__ == "__main__":
    train()