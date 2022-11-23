import dataclasses
from typing import Tuple

from segmentation_voxel.config.config import ModelConfig, TrainingConfig


@dataclasses.dataclass()
class SNEMICOnfig:
    path: str = "/media/brani/DATA/DATASETS/snemi"
    num_classes: int = 2
    classes: Tuple[str] = ("background", "nucleus")
    device: str = "cuda"
    model_config: ModelConfig = ModelConfig()
    training_config: TrainingConfig = TrainingConfig()