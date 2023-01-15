import glob
import os

import h5py
import numpy as np
import torch
import torch.utils.data as data
from segmentation_voxel.config.config import Config

from segmentation_voxel.config.mode import DataMode


class NUCMMLoader(data.Dataset):

    def __init__(self, volumetric_data, labels) -> None:
        super().__init__()
        self.data = volumetric_data
        self.labels = labels
        assert len(self.data) == len(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        label = self.labels[item]
        label = np.array(h5py.File(label, 'r')["main"], dtype=int)
        label = self._get_label(label)
        sample = np.array(h5py.File(sample, "r")["main"], dtype=np.float32)
        return (torch.from_numpy(sample) / 255).unsqueeze(0).float(), torch.from_numpy(label).long()

    def _get_label(self, label):
        label = label > 0
        return label

def get_data_loaders(config: Config, dataset_class: NUCMMLoader = NUCMMLoader):
    path = config.path
    train_data = sorted(glob.glob(os.path.join(path, "Image", "train", "*.h5")))
    eval_data = sorted(glob.glob(os.path.join(path, "Image", "val", "*.h5")))
    train_label = sorted(glob.glob(os.path.join(path, "Label", "train", "*.h5")))
    eval_label = sorted(glob.glob(os.path.join(path, "Label", "val", "*.h5")))


    return {
        DataMode.train: data.DataLoader(
            dataset_class(train_data, train_label),
            batch_size=config.training_config.batch_size,
            shuffle=config.training_config.shuffle[DataMode.train],
            num_workers=config.training_config.num_workers
        ),
        DataMode.eval: data.DataLoader(
            dataset_class(eval_data, eval_label),
            batch_size=config.training_config.batch_size,
            shuffle=config.training_config.shuffle[DataMode.eval],
            num_workers=config.training_config.num_workers
        )
    }


if __name__ == "__main__":
    loaders = get_data_loaders(Config())
    print(loaders)
