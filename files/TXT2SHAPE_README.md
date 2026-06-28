# Text-to-Shape (txt2shape) — 单类别 chair (03001627)

在 SDFusion_mini 框架下新增的文本条件生成模块，沿用原版 SDFusion 的设计：
冻结的 BERT 编码文本 → cross-attention 注入 3D UNet → 在 VQVAE 连续 latent 上做
latent diffusion → classifier-free guidance。

## 文件清单与放置位置

把各文件按相同目录结构放进 SDFusion_mini 项目根目录：

    SDFusion_mini/
    ├── chair_config.py                 # chair VQVAE 配置（训 VQVAE 时已建，txt2shape 复用）
    ├── text_encoder.py                 # 冻结 BERT(bert-base-uncased, 768维)
    ├── text_dataset.py                 # chair 的 (SDF, caption) 配对数据集
    ├── train_txt2shape.py              # 训练脚本（含 CFG dropout + 原子存盘）
    ├── sample_txt2shape.py             # 采样脚本（文本→mesh, DDIM + CFG）
    ├── inspect_txt2shape_ckpt.py       # 读官方权重反推 UNet 配置（可选，对齐官方用）
    ├── diffusion_unet/
    │   ├── __init__.py                 # 覆盖：多导出 TextConditionedUNet3D
    │   └── text_model.py               # TextConditionedUNet3D（cross-attention 版 UNet）
    └── preprocess/
        └── create_chair_text_split.py  # 从 text2shape captions 过滤 chair 并切 train/test

## 依赖

比原项目多一个 transformers（BERT）：

    pip install transformers

首次运行会自动下载 bert-base-uncased（约 400MB）。机器无外网的话，先在有网处
执行 from transformers import BertModel; BertModel.from_pretrained("bert-base-uncased")
缓存好，再把 ~/.cache/huggingface 拷过去。

## 完整运行流程

### 0. 前置：训练好的 chair VQVAE
txt2shape 复用单类 chair VQVAE：checkpoints/vqvae_chair_best.pth，配置来自 chair_config.py。
diffusion 在 VQVAE 的连续 latent (3,16,16,16) 上进行，VQVAE 全程冻结。

### 1. 准备文本数据
    mkdir -p data/ShapeNet/text2shape
    wget http://text2shape.stanford.edu/dataset/captions.tablechair.csv -P data/ShapeNet/text2shape
    python preprocess/create_chair_text_split.py --dataroot data
生成 data/ShapeNet/text2shape/captions.tablechair_{train,test}.csv（内容仅 chair）。
可先验证数据集配对：python text_dataset.py

### 2. 训练
    python train_txt2shape.py
- checkpoint 原子化写到 checkpoints/txt2shape_chair.pth（中断不会写坏）
- 关键超参在脚本顶部：batch_size / num_epochs / lr / p_uncond
- p_uncond=0.1：训练时 10% 概率丢弃文本，采样时才能用 classifier-free guidance

### 3. 采样（文本生成形状）
    python sample_txt2shape.py \
        --prompt "a tall office chair with wheels and armrests" \
        --num_samples 4 --steps 100 --guidance_scale 3.0
- 生成的 .obj 在 outputs/，拖进 MeshLab / Blender 查看
- guidance_scale 越大越贴合文本（常用 3~7.5），设 1.0 关闭 CFG
- steps 是 DDIM 推理步数，100 即可

## 怎么判断训得好不好
txt2shape 的 loss 是噪声预测 MSE，从 ~1.0 缓慢下降，没有直观物理意义。
判断质量主要靠：loss 平稳后直接采样几个 prompt，看椅子像不像、跟文字搭不搭。
chair 单类 + 好的 VQVAE，通常几十个 epoch 就能出像样结果。

## 与原版 SDFusion 的对应关系
- 文本编码器：HF BERT bert-base-uncased(768)，冻结        —— 同
- 条件注入：UNet cross-attention(SpatialTransformer)     —— context_dim=768
- latent：VQVAE 连续 latent                              —— forward_no_quant=True, encode_only=True
- 引导：classifier-free guidance                          —— 训练 dropout + 采样 uncond+s*(cond-uncond)
- 采样器：DDIM                                            —— diffusers.DDIMScheduler
- 扩散步数：1000(标准 DDPM)                               —— num_train_timesteps=1000

## 唯一待对齐项：UNet 超参
VQVAE 已靠官方 yaml 对齐到逐参数一致。但 txt2shape 的 UNet 超参
(model_channels / channel_mult / num_heads / transformer_depth) 目前用的是 mini 项目
ClassConditionedUNet3D 那套数值 + 打开 cross-attention，不是从官方 txt2shape yaml 抄的。

要对齐官方，用 inspect_txt2shape_ckpt.py 读官方权重：
    mkdir -p saved_ckpt
    wget https://uofi.box.com/shared/static/vyqs6aex3rwbgxweyl3qh21c8p6vu33f.pth -O saved_ckpt/sdfusion-txt2shape.pth
    python inspect_txt2shape_ckpt.py --ckpt saved_ckpt/sdfusion-txt2shape.pth
把打印出的 model_channels / context_dim / channel_mult / transformer_depth 按官方数值
改进 diffusion_unet/text_model.py 即可。不改也能正常训练和生成。

## 备注
- diffusion 用 num_train_timesteps=1000（标准 DDPM），独立于 train_diffusion.py（class 条件，5000 步），互不影响。
- 单类 chair 数据量足够时建议多训几十个 epoch，loss 不再明显下降即可采样观察。
