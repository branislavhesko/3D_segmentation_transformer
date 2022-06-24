from msilib import AMD64
import torch
from tqdm import tqdm

from segmentation_3d.config.config import Config
from segmentation_3d.config.mode import DataMode
from segmentation_3d.data.dataset_nucmm import get_data_loaders


class SegmentationTrainer:
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.data_loaders = get_data_loaders(config)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.training_config.learning_rate, 
            weight_decay=0.0001, 
            amsgrad=True)
    
    def train(self):
        
        for idx in range(self.config.training_config.num_epochs):
            pass
        
    def _train_epoch(self):
        loader = tqdm(self.data_loaders[DataMode.train])
        for data in enumerate(loader):
            pass
    
    def _validate_epoch(self):
        pass