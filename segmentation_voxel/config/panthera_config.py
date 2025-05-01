import dataclasses
from typing import Callable
from segmentation_voxel.config.config import TrainingConfig, ModelConfig


def _default_transforms(x, y):
    return x, y


@dataclasses.dataclass
class PantherConfig:
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    num_classes: int = 2
    classes: list[str] = ("background", "cancer")
    device: str = "cuda"
    path: str = "/home/brani/code/3D_segmentation_transformer/data/PANTHER_Task1/"
    masks_subfolder: str = "LabelsTr"
    images_subfolder: str = "ImagesTr"
    transforms: Callable = _default_transforms
    padded_shape: tuple[int, int, int] = (80, 260, 320)
    