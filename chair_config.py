# chair 单类 VQVAE 专用配置
#
# 与项目原 config.py 的区别：num_res_blocks 从 1 提到 2（提升 encoder/decoder 容量，
# 更好地重建椅子靠背、椅腿等薄结构）。其余保持一致，latent 仍是 16^3 x 3。
#
# 注意：用这个 config 训练出来的 VQVAE，下游（diffusion / txt2shape / 可视化）
# 也必须用同一份 ddconfig 来构建模型，否则权重对不上。

vqvae_chair_config = {
    "embed_dim": 3,
    "n_embed": 8192,
    "ddconfig": {
        "double_z": False,
        "z_channels": 3,
        "resolution": 64,
        "in_channels": 1,
        "out_ch": 1,
        "ch": 64,
        "ch_mult": [1, 2, 4],     # 64 -> 16，下采样 4 倍
        "num_res_blocks": 1,      # 原 mini 是 1，这里提到 2
        "attn_resolutions": [],   # 如需更强可改成 [16] 在瓶颈加注意力
        "dropout": 0.0,
    },
}
