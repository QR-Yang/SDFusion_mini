"""
inspect_txt2shape_ckpt.py — 从官方 SDFusion txt2shape 权重反推 UNet 真实配置

先下载官方权重（README 提供的链接）：
    mkdir -p saved_ckpt
    wget https://uofi.box.com/shared/static/vyqs6aex3rwbgxweyl3qh21c8p6vu33f.pth \
         -O saved_ckpt/sdfusion-txt2shape.pth

然后：
    python inspect_txt2shape_ckpt.py --ckpt saved_ckpt/sdfusion-txt2shape.pth

脚本会从 state_dict 的张量形状里读出 UNet 的：
    in/out_channels, model_channels, context_dim(文本编码维度),
    channel 进度(→ channel_mult), input/output_blocks 数(→ num_res_blocks),
    transformer_depth, 以及 UNet 部分的参数量。

num_heads 无法从权重形状直接读出（它只是 reshape 参数），
需要时以官方 yaml 为准，或用 LDM 默认 8 / num_head_channels=32。
"""

import argparse
import re
from collections import defaultdict, OrderedDict

import torch


def load_state(path):
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in raw and isinstance(raw[k], dict):
                raw = raw[k]
                break
    return raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--show_unet_keys", action="store_true",
                   help="打印 UNet 部分全部 key 和形状")
    args = p.parse_args()

    state = load_state(args.ckpt)

    # ── 找到 UNet 子模块的 key 前缀 ───────────────────────────
    # 官方可能把 unet 存成 df.xxx / model.diffusion_model.xxx / unet.xxx
    # 统一用“包含 input_blocks”来定位
    unet_keys = [k for k in state.keys() if "input_blocks" in k
                 or "middle_block" in k or "output_blocks" in k or "time_embed" in k]
    if not unet_keys:
        print("[!] 没找到 UNet（input_blocks/middle_block）相关的 key。")
        print("    先看看顶层 key 长什么样：")
        for k in list(state.keys())[:20]:
            print("   ", k, tuple(state[k].shape) if hasattr(state[k], "shape") else "")
        return

    # 推断公共前缀（input_blocks 之前的部分）
    sample = next(k for k in unet_keys if "input_blocks.0.0.weight" in k)
    prefix = sample[: sample.index("input_blocks")]
    print(f"\n{'='*56}")
    print(f"  ckpt: {args.ckpt}")
    print(f"  UNet 前缀: '{prefix}'")
    print(f"{'='*56}\n")

    def g(name):
        k = prefix + name
        return state[k] if k in state else None

    unet_params = sum(v.numel() for k, v in state.items()
                      if k.startswith(prefix) and hasattr(v, "numel"))
    print(f"[UNet 参数量] {unet_params/1e6:.2f} M\n")

    # conv_in: (model_channels, in_channels, 3,3,3)
    w = g("input_blocks.0.0.weight")
    if w is not None:
        print(f"[推断] model_channels = {w.shape[0]}")
        print(f"[推断] in_channels    = {w.shape[1]}")

    # conv_out: 常见为 out.2.weight 或 out.0.weight
    for cand in ("out.2.weight", "out.0.weight", "out.weight"):
        w = g(cand)
        if w is not None:
            print(f"[推断] out_channels   = {w.shape[0]}  (from {cand})")
            break

    # context_dim：cross-attention 的 to_k / to_v 把 context 投影到 inner_dim
    #   to_k.weight 形状 = (inner_dim, context_dim)
    ctx = None
    for k in state.keys():
        if k.startswith(prefix) and re.search(r"attn2\.to_k\.weight$", k):
            ctx = state[k].shape[1]
            inner = state[k].shape[0]
            print(f"[推断] context_dim    = {ctx}   "
                  f"(文本编码维度；768=HF bert-base, 1280=LDM BERT)")
            print(f"        cross-attn inner_dim = {inner}  (= n_heads * dim_head)")
            break
    if ctx is None:
        print("[!] 没找到 attn2.to_k（可能这个权重不含 cross-attention 文本条件）")

    # transformer_depth：单个 SpatialTransformer 里 transformer_blocks 的层数
    depths = defaultdict(set)
    for k in state.keys():
        m = re.search(r"transformer_blocks\.(\d+)\.", k)
        if m and k.startswith(prefix):
            # 用所在 block 路径做 key
            base = k[: m.start()]
            depths[base].add(int(m.group(1)))
    if depths:
        td = max(len(v) for v in depths.values())
        print(f"[推断] transformer_depth = {td}")

    # input_blocks 数量 + 通道进度 → channel_mult / num_res_blocks
    in_block_ids = set()
    block_out_ch = OrderedDict()
    for k in state.keys():
        if not k.startswith(prefix):
            continue
        m = re.search(r"input_blocks\.(\d+)\.", k)
        if m:
            idx = int(m.group(1))
            in_block_ids.add(idx)
            # resblock 的输出通道：in_layers/out_layers 里的 conv 权重
            mm = re.search(r"input_blocks\.\d+\.0\.out_layers\.\d+\.weight$", k)
            if mm and state[k].dim() >= 1:
                block_out_ch[idx] = state[k].shape[0]
    n_in = max(in_block_ids) + 1 if in_block_ids else 0
    print(f"\n[推断] input_blocks 数量 = {n_in}")
    if block_out_ch:
        ch_seq = [block_out_ch.get(i) for i in sorted(block_out_ch)]
        mc = g("input_blocks.0.0.weight")
        mc = mc.shape[0] if mc is not None else None
        print(f"[推断] 各 input_block 输出通道 = {ch_seq}")
        if mc:
            mults = sorted(set(c // mc for c in ch_seq if c))
            print(f"[推断] model_channels = {mc}  → channel_mult 里的倍数集合 ≈ {mults}")
    print("\n  （把上面这些数贴给我，我据此把 TextConditionedUNet3D 调成和官方一致）")

    if args.show_unet_keys:
        print(f"\n[UNet 全部 key]")
        for k in sorted(k for k in state.keys() if k.startswith(prefix)):
            print(f"  {k[len(prefix):]:<55s} {tuple(state[k].shape)}")


if __name__ == "__main__":
    main()
