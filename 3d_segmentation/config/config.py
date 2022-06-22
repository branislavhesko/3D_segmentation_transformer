import dataclasses


@dataclasses.dataclass()
class TrainingConfig:
    batch_size: int = 4
    num_workers: int = 4


@dataclasses.dataclass()
class Config:
    path: str = "/home/brani/DATASETS/Zebrafish (NucMM-Z)"
    training_config: TrainingConfig = TrainingConfig()
