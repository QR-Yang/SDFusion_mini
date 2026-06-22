import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
class ShapeNetSDFDataset(Dataset):
    def __init__(self, phase="train"):
        self.root = "data/ShapeNet/SDF_v1/resolution_64"
        self.items = []
        self.add_class("03001627", 0, phase)
        self.add_class("04379243", 1, phase)
        self.add_class("02958343", 2, phase)
        self.add_class("04090263", 3, phase)
        self.add_class("02691156", 4, phase)
    def add_class(self, rawid, class_id, phase):
        list_path = "dataset_info_files/ShapeNet_filelists/" + rawid + "_" + phase + ".lst"
        f = open(list_path)
        model_ids = f.readlines()
        f.close()
        for model_id in model_ids:
            model_id = model_id.strip()
            sdf_path = self.root + "/" + rawid + "/" + model_id + "/ori_sample_grid.h5"
            self.items.append((sdf_path, class_id))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, index):
        sdf_path, class_id = self.items[index]
        h5_file = h5py.File(sdf_path, "r")
        sdf = h5_file["pc_sdf_sample"][:].astype(np.float32)
        h5_file.close()
        sdf = torch.from_numpy(sdf).view(1, 64, 64, 64)
        sdf = torch.clamp(sdf, -0.2, 0.2)
        class_id = torch.tensor(class_id).long()
        return sdf, class_id
def get_dataloader(batch_size=4, phase="train"):
    dataset = ShapeNetSDFDataset(phase=phase)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader
if __name__ == "__main__":
    dataset = ShapeNetSDFDataset()
    x, class_id = dataset[0]