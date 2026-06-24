import os
os.environ["TORCH_BLAS_PREFER_CUBLASLT"] = "0"
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"
import torch
import torch.nn.functional as F
from tqdm import tqdm
from config import vqvae_config
from datasets import get_dataloader
from vqvae import VQVAE
batch_size = 12
num_epochs = 30
lr = 1e-4
save_path = "checkpoints/vqvae_1e-4.pth"
def train():
    device = "cuda"
    dataloader = get_dataloader(batch_size=batch_size)
    model = VQVAE(
        ddconfig=vqvae_config["ddconfig"],
        n_embed=vqvae_config["n_embed"],
        embed_dim=vqvae_config["embed_dim"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        total_loss = 0
        for data in tqdm(dataloader):
            sdf = data[0].to(device)
            recon, codebook_loss = model(sdf)
            recon_loss = F.l1_loss(recon, sdf)
            loss = recon_loss + codebook_loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("epoch:", epoch + 1, "loss:", total_loss / len(dataloader))
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), save_path)
if __name__ == "__main__":
    train()
