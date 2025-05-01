import dataclasses
import pathlib
from typing import Callable

import SimpleITK as sitk
import torch
import torch.nn.functional as F
import numpy as np

from segmentation_voxel.config.panthera_config import PantherConfig


@dataclasses.dataclass
class PantherAnnotation:
    image_path: str
    mask_path: str


def mha_to_tensor(mha_path, normalize=True):
    """
    Convert an MHA file to a PyTorch tensor.
    
    Args:
        mha_path (str): Path to the MHA file
        
    Returns:
        torch.Tensor: The image data as a tensor
    """
    # Read the MHA file
    image = sitk.ReadImage(mha_path)
    
    # Convert to numpy array
    np_array = np.asarray(sitk.GetArrayFromImage(image)).astype(np.float32)
    print(np_array.shape)
    tensor = torch.from_numpy(np_array).float()
    
    # Normalize to [0, 1] range if needed
    if normalize and tensor.max() > 1.0:
        tensor = tensor / tensor.max()
    
    return tensor



def load_annotations(config: PantherConfig):
    path = pathlib.Path(config.path)
    images_subfolder = path / config.images_subfolder
    masks_subfolder = path / config.masks_subfolder
    annotations = []
    for image_path in images_subfolder.glob("*.mha"):
        mask_path = masks_subfolder / str(image_path.name).replace("_0000.mha", ".mha")
        annotations.append(PantherAnnotation(image_path, mask_path))
    return annotations



class PantherDataset(torch.utils.data.Dataset):
    def __init__(self, config: PantherConfig, annotations: list[PantherAnnotation], transforms: Callable):
        self.annotations = annotations
        self.transforms = transforms
        self.config = config
    def __len__(self):
        return len(self.annotations)
    
    def _pad_volume(self, volume, padded_shape):
        c, h, w = volume.shape
        pad_w = (w - padded_shape[2]) // 2
        pad_h = (h - padded_shape[1]) // 2
        pad_c = (c - padded_shape[0]) // 2
        return F.pad(volume, (pad_w, pad_w, pad_h, pad_h, pad_c, pad_c), "constant", 0)
    
    def __getitem__(self, index):
        image_path = self.annotations[index].image_path
        mask_path = self.annotations[index].mask_path
        image = mha_to_tensor(image_path)
        mask = mha_to_tensor(mask_path)
        image, mask = self._pad_volume(image, self.config.padded_shape), self._pad_volume(mask, self.config.padded_shape)
        if self.transforms:
            image, mask = self.transforms(image, mask)
        return image, mask
    
    
def get_dataloaders(config: PantherConfig):
    annotations = load_annotations(config)
    dataset = PantherDataset(config, annotations, config.transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )
    return dataloader


if __name__ == "__main__":
    config = PantherConfig()
    dataloader = get_dataloaders(config)
    for batch in dataloader:
        print(batch)
        break
