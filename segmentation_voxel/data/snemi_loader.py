import glob
import os
import tifffile

import numpy as np
import torch
import torch.utils.data as data

from segmentation_voxel.config.config import Config, DataMode


class SnemiDataset(data.Dataset):
    # TODO: add stride
    def __init__(self, data, labels, patch_size):
        self.data = data
        self.labels = labels.squeeze()
        self.labels[self.labels < 20] = 0
        self.labels[self.labels >= 20] = 1
        self.patch_size = patch_size

    def __getitem__(self, index):
        position = (
            torch.randint(0, self.data.shape[1] - self.patch_size[0] - 1, (1,)),
            torch.randint(0, self.data.shape[2] - self.patch_size[1] - 1, (1,)),
            torch.randint(0, self.data.shape[3] - self.patch_size[2] - 1, (1,)))
        data = self.data[
            :,
            position[0]:position[0] + self.patch_size[0],
            position[1]:position[1] + self.patch_size[1],
            position[2]:position[2] + self.patch_size[2]
        ]
        labels = self.labels[
            position[0]:position[0] + self.patch_size[0],
            position[1]:position[1] + self.patch_size[1],
            position[2]:position[2] + self.patch_size[2]
        ]
        # TODO: add augmentation
        return data, labels

    def __len__(self):
        return 64

    def __repr__(self):
        return f'{self.__class__.__name__}({len(self)})'

    def __str__(self):
        return self.__repr__()


def load_data(train_data_path, train_mask_path, test_data_path):
    test = _load_tif_data(test_data_path)
    data = _load_tif_data(train_data_path)
    mask = _load_tif_data(train_mask_path)
    return data, mask, test

def _load_tif_data(data_path):
    images = []
    with tifffile.TiffFile(data_path) as tif:
        for page in tif.pages:
            image = page.asarray()
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255
                images.append(torch.from_numpy(image).float())
            elif image.dtype == np.uint16:
                image = image.astype(np.int32)
                images.append(torch.from_numpy(image).long())
    images = torch.stack(images).unsqueeze(0)
    return images


def get_data_loaders_snemi(config: Config):
    path = config.path

    train, mask, test = load_data(
        train_data_path=os.path.join(path, "image", 'train-input.tif'),
        train_mask_path=os.path.join(path, "seg", 'train-labels.tif'),
        test_data_path=os.path.join(path, "image", 'test-input.tif')
    )

    return {
        DataMode.train: data.DataLoader(
            SnemiDataset(train, mask, patch_size=config.model_config.input_shape[1:]),
            batch_size=config.training_config.batch_size,
            shuffle=config.training_config.shuffle[DataMode.train],
            num_workers=config.training_config.num_workers
        ),
        DataMode.eval: data.DataLoader(
            SnemiDataset(test, torch.zeros_like(test).long(), patch_size=config.model_config.input_shape[1:]),
            batch_size=config.training_config.batch_size,
            shuffle=config.training_config.shuffle[DataMode.eval],
            num_workers=config.training_config.num_workers
        )
    }