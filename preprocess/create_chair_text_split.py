"""
create_chair_text_split.py — 从 text2shape 的 captions.tablechair.csv
过滤出 chair (03001627) 的 caption，并切分成 train/test。

输出：
    data/ShapeNet/text2shape/captions.tablechair_train.csv
    data/ShapeNet/text2shape/captions.tablechair_test.csv
（沿用项目里 text_dataset.py 读取的文件名，里面同时含 train/test，
  ChairTextDataset 会在读取时再过滤 chair。）

如果你只想要 chair，也可以用 --chair_only 额外输出纯 chair 的文件：
    captions.chair_train.csv / captions.chair_test.csv

用法（在项目根目录或 preprocess 目录下都行）：
    python preprocess/create_chair_text_split.py --dataroot data
"""

import os
import csv
import argparse
import numpy as np

CHAIR_SYNSET = "03001627"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="data")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--chair_only", action="store_true",
                        help="额外输出仅含 chair 的 captions.chair_{train,test}.csv")
    opt = parser.parse_args()

    csv_file = f"{opt.dataroot}/ShapeNet/text2shape/captions.tablechair.csv"
    assert os.path.exists(csv_file), f"找不到 {csv_file}，请先下载 text2shape captions。"

    np.random.seed(opt.seed)

    with open(csv_file, newline="") as f:
        reader = csv.reader(f, delimiter=",")
        header = next(reader, None)
        data = [row for row in reader]

    # 找出 topLevelSynsetId 所在列，用来过滤 chair
    syn_idx = None
    if header is not None:
        for i, col in enumerate(header):
            if col.strip() == "topLevelSynsetId":
                syn_idx = i
                break

    def is_chair(row):
        if syn_idx is not None and syn_idx < len(row):
            return row[syn_idx].strip() == CHAIR_SYNSET
        # 退化：用 category 列（通常是第 4 列）判断
        return any(c.strip().lower() == "chair" for c in row)

    chair_data = [r for r in data if is_chair(r)]
    print(f"[*] 全部 caption {len(data)} 行，其中 chair {len(chair_data)} 行")

    # 切分 train/test（按 caption 行切分，简单直接）
    np.random.shuffle(chair_data)
    n_train = int(len(chair_data) * opt.train_ratio)
    train_data = chair_data[:n_train]
    test_data = chair_data[n_train:]

    out_dir = f"{opt.dataroot}/ShapeNet/text2shape"
    os.makedirs(out_dir, exist_ok=True)

    # 1) 沿用项目命名（captions.tablechair_{phase}.csv），内容是 chair-only，
    #    ChairTextDataset 读取时会再次确认 chair，所以这里直接写 chair 数据即可。
    for phase, d in [("train", train_data), ("test", test_data)]:
        out_csv = os.path.join(out_dir, f"captions.tablechair_{phase}.csv")
        with open(out_csv, "wt", newline="") as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(header)
            w.writerows(d)
        print(f"[*] 写出 {out_csv}：{len(d)} 行")

    # 2) 可选：额外输出明确命名的 chair-only 文件
    if opt.chair_only:
        for phase, d in [("train", train_data), ("test", test_data)]:
            out_csv = os.path.join(out_dir, f"captions.chair_{phase}.csv")
            with open(out_csv, "wt", newline="") as f:
                w = csv.writer(f, delimiter=",")
                w.writerow(header)
                w.writerows(d)
            print(f"[*] 写出 {out_csv}：{len(d)} 行")


if __name__ == "__main__":
    main()
