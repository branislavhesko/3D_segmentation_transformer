import torch
from tqdm import tqdm
import vedo
from vedo import Volume
from vedo.applications import RayCastPlotter

from segmentation_voxel.config.config import Config
from segmentation_voxel.config.mode import DataMode
from segmentation_voxel.data.dataset_nucmm import get_data_loaders
from segmentation_voxel.modeling.voxel_former import SegmentationTransformer3D
from segmentation_voxel.utils.metrics import Metrics


class SegmentationTrainer:
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.data_loaders = get_data_loaders(config)

        num_patches = self.config.model_config.input_shape[1] * self.config.model_config.input_shape[2] * self.config.model_config.input_shape[3] // (self.config.model_config.patch_size ** 3)
        self.model = SegmentationTransformer3D(
            num_classes=self.config.num_classes,
            embed_size=self.config.model_config.embed_size,
            num_heads=self.config.model_config.num_heads,
            input_channels=self.config.model_config.input_channels,
            channels=8,
            patch_size=self.config.model_config.patch_size,
            input_shape=num_patches,
            dropout=self.config.training_config.dropout_probability
        ).to(self.config.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.training_config.learning_rate, 
            weight_decay=0.0001, 
            amsgrad=True)
        self.loss = torch.nn.CrossEntropyLoss()
        self.metrics = {
            DataMode.train: Metrics(classes=self.config.classes),
            DataMode.eval: Metrics(classes=self.config.classes)
        }
    
    def train(self):
        
        for epoch in range(self.config.training_config.num_epochs):
            self._train_epoch(epoch)
            
            if epoch % self.config.training_config.validation_frequency == 0:
                self._validate_epoch(epoch)
        
    def _train_epoch(self, epoch):
        self.model.train()
        loader = tqdm(self.data_loaders[DataMode.train])
        for index, data in enumerate(loader):
            self.optimizer.zero_grad()
            image, label = [d.to(self.config.device) for d in data]
            model_output = self.model(image)
            prediction = model_output.argmax(dim=1)
            loss = self.loss(model_output, label)
            loss.backward()
            self.optimizer.step()
            loader.set_description(f"EPOCH: {epoch}, Loss: {loss.item():.4f}")
            self.metrics[DataMode.train].update(prediction, label)
        if self.config.training_config.live_visualization:
            self._visualize(model_output)
            
        print(f"Train Metrics: {self.metrics[DataMode.train]}")

    def _visualize(self, model_output):
        vedo.close()
        vol = Volume(model_output[0, 0, :, :, :].sigmoid().detach().cpu().numpy())
        plt = RayCastPlotter(vol, bg='black', bg2='blackboard', axes=7, interactive=True)  # Plotter instance
        plt.escaped = False
        plt.show(interactive=True).close()

    @torch.no_grad()
    def _validate_epoch(self, epoch):
        self.model.eval()
        loader = tqdm(self.data_loaders[DataMode.eval])
        
        for index, data in enumerate(loader):
            volume, label = [d.to(self.config.device) for d in data]
            model_output = self.model(volume)
            prediction = model_output.argmax(dim=1)
            loss = self.loss(model_output, label)
            self.metrics[DataMode.eval].update(prediction, label)
        
        if self.config.training_config.live_visualization:
            self._visualize(model_output)
        print(f"Validation Metrics: {self.metrics[DataMode.eval]}")    

if __name__ == "__main__":
    SegmentationTrainer(Config()).train()
