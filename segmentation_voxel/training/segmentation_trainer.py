from msilib import AMD64
import torch
from tqdm import tqdm

from segmentation_voxel.config.config import Config
from segmentation_voxel.config.mode import DataMode
from segmentation_voxel.data.dataset_nucmm import get_data_loaders
from segmentation_voxel.modeling.voxel_former import SegmentationTransformer3D


class SegmentationTrainer:
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.data_loaders = get_data_loaders(config)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.training_config.learning_rate, 
            weight_decay=0.0001, 
            amsgrad=True)
        num_patches = self.config.model_config.input[1] * self.config.model_config.input_shape[2] * self.config.model_config.input_shape[3] // (self.config.model_config.patch_size ** 3)
        self.model = SegmentationTransformer3D(
            num_classes=self.config.num_classes,
            embed_size=self.config.model_config.embed_size,
            num_heads=self.config.model_config.num_heads,
            input_channels=self.config.model_config.input_channels,
            patch_size=self.config.model_config.patch_size,
            input_shape=num_patches,
            dropout=self.config.training_config.dropout_probability
        )
    
    def train(self):
        
        for idx in range(self.config.training_config.num_epochs):
            pass
        
    def _train_epoch(self):
        loader = tqdm(self.data_loaders[DataMode.train])
        for data in enumerate(loader):
            pass
    
    def _validate_epoch(self):
        pass