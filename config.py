vqvae_config = {
    "embed_dim": 3,
    "n_embed": 8192,
    "ddconfig": {
        "double_z": False,
        "z_channels": 3,
        "resolution": 64,
        "in_channels": 1,
        "out_ch": 1,
        "ch": 64,
        "ch_mult": [1, 2, 4],
        "num_res_blocks": 1,
        "attn_resolutions": [],
        "dropout": 0.0,
    },
}
