from tqdm import tqdm
import torch

from segmentation_voxel.config.mode import DataMode
from segmentation_voxel.modeling.instance_former import InstanceSegmentationFormer3D
from segmentation_voxel.modeling.instance_loss import InstanceLoss
from segmentation_voxel.data.dataset_instance_nucmm import InstanceNucMM
from segmentation_voxel.data.dataset_nucmm import get_data_loaders


class InstanceTrainer:

    def __init__(self, config) -> None:
        self.config = config
        self.model = InstanceSegmentationFormer3D(
            num_classes=config.num_classes,
            num_queries=config.num_queries,
            embed_size=config.embed_size,
            num_heads=config.num_heads,
            input_channels=config.input_channels,
            channels=config.channels,
            patch_size=config.patch_size,
            input_shape=config.input_shape,
            dropout=config.dropout,
        ).to(self.config.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.loss = InstanceLoss()
        self.data_loaders = get_data_loaders(config, InstanceNucMM)

    def train(self):
        for epoch in range(self.config.epochs):
            self.train_epoch()
            self.validate_epoch()

    def train_epoch(self, epoch):
        self.model.train()
        loader = tqdm(self.data_loaders[DataMode.train])
        for idx, data in enumerate(loader):
            self.optimizer.zero_grad()
            image, label = [d.to(self.config.device) for d in data]
            model_output = self.model(image)
            loss = self.loss(model_output, label)
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()
        loader = tqdm(self.data_loaders[DataMode.eval])

        for idx, data in enumerate(loader):
            image, label = [d.to(self.config.device) for d in data]
            model_output = self.model(image)
            loss = self.loss(model_output, label)
