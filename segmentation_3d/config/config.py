import dataclasses

from segmentation_3d.config.mode import DataMode


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

@dataclasses.dataclass()
class Config:
    path: str = "/home/brani/DATASETS/Zebrafish (NucMM-Z)"
    training_config: TrainingConfig = TrainingConfig()
