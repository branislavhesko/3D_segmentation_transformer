import glob
import os

import torch.utils.data as data

from segmentation_3d.config.mode import DataMode


class NUCMMLoader(data.Dataset):
    
    def __init__(self) -> None:
        super().__init__()
        
    def __len__(self):
        pass
    
    def __getitem__(self, item):
        pass


def get_data_loaders(config):
    path = config.path
    train_data = sorted(glob.glob(os.path.join(path, "Image", "train", "*.h5")))
    eval_data = sorted(glob.glob(os.path.join(path, "Image", "val", "*.h5")))
    train_label = sorted(glob.glob(os.path.join(path, "Label", "train", "*.h5")))
    eval_label = sorted(glob.glob(os.path.join(path, "Label", "val", "*.h5")))
    
    
    return {
        DataMode.train: data.DataLoader(
            NUCMMLoader(),
        ),
        DataMode.eval: data.DataLoader(
        )
    }
