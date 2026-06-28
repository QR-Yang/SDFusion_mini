"""
text_dataset.py — text2shape 文本-SDF 配对数据集（单类别 chair / 03001627）

数据来源：
  - SDF：   data/ShapeNet/SDF_v1/resolution_64/03001627/<model_id>/ori_sample_grid.h5
  - 文本：  data/ShapeNet/text2shape/captions.tablechair_<phase>.csv

text2shape 的 caption CSV 表头通常为：
  id, modelId, description, category, topLevelSynsetId, subSynsetId

每个 chair 模型可能对应多条 caption，这里把 (caption, model_id) 当作一条样本，
即同一个形状会以不同文字描述出现多次，这是 text2shape 的标准做法。
"""

import os
import csv
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

CHAIR_SYNSET = "03001627"
SDF_ROOT = "data/ShapeNet/SDF_v1/resolution_64"
CAPTION_DIR = "data/ShapeNet/text2shape"


def _clean_text(s: str) -> str:
    """去掉 text2shape caption 里常见的首尾引号和多余空白。"""
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s.strip()


class ChairTextDataset(Dataset):
    def __init__(self, phase="train", sdf_root=SDF_ROOT, caption_dir=CAPTION_DIR):
        self.sdf_root = sdf_root
        self.synset = CHAIR_SYNSET

        csv_path = os.path.join(caption_dir, f"captions.tablechair_{phase}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"找不到 caption 文件: {csv_path}\n"
                f"请先运行 preprocess/create_chair_text_split.py 生成 train/test split。"
            )

        self.items = []           # list of (sdf_path, caption)
        n_total, n_kept = 0, 0
        missing_sdf = 0

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n_total += 1
                model_id = (row.get("modelId") or "").strip()
                desc     = _clean_text(row.get("description") or "")
                category = (row.get("category") or "").strip().lower()
                synset   = (row.get("topLevelSynsetId") or "").strip()

                if not model_id or not desc:
                    continue
                # 只保留 chair：优先用 synset 判断，退化到 category 名
                is_chair = (synset == CHAIR_SYNSET) or (category == "chair")
                if not is_chair:
                    continue

                sdf_path = os.path.join(
                    self.sdf_root, self.synset, model_id, "ori_sample_grid.h5"
                )
                if not os.path.isfile(sdf_path):
                    missing_sdf += 1
                    continue

                self.items.append((sdf_path, desc))
                n_kept += 1

        print(f"[ChairTextDataset/{phase}] caption 总行数 {n_total}，"
              f"保留 chair 配对 {n_kept}（缺 SDF 跳过 {missing_sdf}）")

        if len(self.items) == 0:
            raise RuntimeError(
                "没有任何有效的 (文本, SDF) 配对。请检查：\n"
                f"  1. SDF 是否在 {self.sdf_root}/{self.synset}/<model_id>/ori_sample_grid.h5\n"
                f"  2. caption CSV 是否包含 chair 类别"
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        sdf_path, caption = self.items[index]
        with h5py.File(sdf_path, "r") as f:
            sdf = f["pc_sdf_sample"][:].astype(np.float32)
        sdf = torch.from_numpy(sdf).view(1, 64, 64, 64)
        sdf = torch.clamp(sdf, -0.2, 0.2)   # 与 datasets.py / VQVAE 训练保持一致
        return sdf, caption


def get_text_dataloader(batch_size=8, phase="train", num_workers=4, shuffle=None):
    dataset = ChairTextDataset(phase=phase)
    if shuffle is None:
        shuffle = (phase == "train")
    # caption 是字符串，默认 collate 会把它们组成 list/tuple，正合适
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader


if __name__ == "__main__":
    ds = ChairTextDataset(phase="train")
    sdf, cap = ds[0]
    print("sdf:", sdf.shape, "| caption:", cap)
