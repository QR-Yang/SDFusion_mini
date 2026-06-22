import torch.nn as nn

from models.networks.diffusion_networks.openai_model_3d import UNet3DModel


class ClassConditionedUNet3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.unet = UNet3DModel(
            image_size=16,
            in_channels=3,
            out_channels=3,
            model_channels=192,
            num_res_blocks=2,
            attention_resolutions=[1, 2, 4],
            channel_mult=[1, 2, 4, 4],
            num_heads=6,
            dims=3,
            num_classes=5,
        )

    def forward(self, x, t, class_id):
        x = self.unet(x, t, y=class_id)
        return x
