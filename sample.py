import os
os.environ["TORCH_BLAS_PREFER_CUBLASLT"] = "0"
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"
import torch
import trimesh
from diffusers import DDPMScheduler
from skimage import measure
from config import vqvae_config
from diffusion_unet import ClassConditionedUNet3D
from vqvae import VQVAE
class_name = "airplane"
num_samples = 4
num_train_timesteps = 5000
num_inference_steps = 5000
vqvae_path = "checkpoints/vqvae.pth"
diffusion_path = "checkpoints/class_diffusion.pth"
save_dir = "outputs"
def get_class_id(name):
    if name == "chair":
        return 0
    if name == "table":
        return 1
    if name == "car":
        return 2
    if name == "rifle":
        return 3
    if name == "airplane":
        return 4
def save_obj(sdf, path):
    sdf = sdf[0]
    verts, faces, normals, values = measure.marching_cubes(sdf, level=0.0)
    verts = verts / 63.0 - 0.5
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(path)
def sample():
    device = "cuda"
    vqvae = VQVAE(
        ddconfig=vqvae_config["ddconfig"],
        n_embed=vqvae_config["n_embed"],
        embed_dim=vqvae_config["embed_dim"],
    ).to(device)
    vqvae.load_state_dict(torch.load(vqvae_path))
    vqvae.eval()
    model = ClassConditionedUNet3D().to(device)
    model.load_state_dict(torch.load(diffusion_path))
    model.eval()
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    scheduler.set_timesteps(num_inference_steps)
    z = torch.randn(num_samples, 3, 16, 16, 16).to(device)
    class_id = torch.tensor([get_class_id(class_name)] * num_samples, device=device).long()
    with torch.no_grad():
        for t in scheduler.timesteps:
            timestep = torch.tensor([t] * num_samples, device=device).long()
            noise_pred = model(z, timestep, class_id)
            z = scheduler.step(noise_pred, t, z).prev_sample
        sdf = vqvae.decode_no_quant(z)
    os.makedirs(save_dir, exist_ok=True)
    sdf = sdf.cpu().numpy()
    for i in range(num_samples):
        save_path = save_dir + "/" + class_name + "_" + str(i) + ".obj"
        save_obj(sdf[i], save_path)
    print("saved to:", save_dir)
if __name__ == "__main__":
    sample()