import torch.nn as nn

from models.networks.diffusion_networks.openai_model_3d import UNet3DModel


class TextConditionedUNet3D(nn.Module):
    """
    文本条件的 3D UNet（用于 txt2shape）。

    与 ClassConditionedUNet3D 的区别：
      - 去掉 num_classes（不再用 class embedding）
      - 打开 use_spatial_transformer，通过 cross-attention 接收文本 context
      - context_dim=768 对应 BERT 的输出维度

    forward 的 context 形状为 (B, L, 768)，由 BERTTextEncoder 提供。
    """

    def __init__(self, context_dim: int = 768):
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
            num_classes=None,              # 不用 class 条件
            use_spatial_transformer=True,  # 打开 cross-attention
            transformer_depth=1,
            context_dim=context_dim,       # BERT = 768
        )

    def forward(self, x, t, context):
        # context: (B, L, 768)
        return self.unet(x, t, context=context)
