import dataclasses
from typing import Tuple

from torch import dropout

from segmentation_voxel.config.mode import DataMode


@dataclasses.dataclass()
class TrainingConfig:
    batch_size: int = 4
    num_workers: int = 4
    shuffle = {
        DataMode.train: True,
        DataMode.eval: False
    }
    learning_rate: float =  0.001
    num_epochs: int = 20
    dropout_probability: float = 0.1
    
@dataclasses.dataclass()
class ModelConfig:
    model: str = "small"
    embed_size: int = 96
    num_heads: int = 8
    input_channels: int = 3
    patch_size: int = 16
    input_shape: Tuple[int, int, int] = (3, 128, 128, 128)

@dataclasses.dataclass()
class Config:
    path: str = "/home/brani/DATASETS/Zebrafish (NucMM-Z)"
    training_config: TrainingConfig = TrainingConfig()
    model_config: ModelConfig = ModelConfig()
    num_classes: int = 2
