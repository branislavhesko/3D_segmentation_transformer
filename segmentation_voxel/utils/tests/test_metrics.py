import pytest
import numpy as np
import torch

from segmentation_voxel.utils.metrics import Metrics


class TestMetrics:
    
    @pytest.fixture(scope="function")
    def metrics(self):
        return Metrics(classes=["background", "class1"])
    
    @pytest.mark.parametrize("prediction, label, expected", [
        (torch.tensor([0, 0, 0, 0, 0]), torch.tensor([0, 0, 0, 0, 0]), (1, np.nan)),
        (torch.tensor([0, 1, 0, 1, 0]), torch.tensor([0, 1, 0, 0, 0]), (0.75, 0.5)),
        (torch.tensor([1, 1, 1, 1, 1]), torch.tensor([0, 0, 0, 0, 0]), (0, 0)),
        (torch.tensor([1, 0, 0, 0, 0]), torch.tensor([1, 0, 0, 0, 0]), (1, 1)),
        (torch.tensor([1, 1, 1, 0, 0]), torch.tensor([1, 1, 0, 0, 0]), (0.66, 0.66))
    ])
    def test_iou(self, prediction, label, expected, metrics):
        for idx, _ in enumerate(metrics.classes):
            if not np.isnan(expected[idx]):
                assert np.allclose(metrics._iou(prediction == idx, label == idx).item(), 
                                   np.array(expected[idx]), atol=1e-2)
            else:
                assert np.isnan(metrics._iou(prediction == idx, label == idx).item())

    @pytest.mark.parametrize("prediction, label, expected", [
        (torch.tensor([0, 0, 0, 0, 0]), torch.tensor([0, 0, 0, 0, 0]), (1, np.nan)),
        (torch.tensor([0, 1, 0, 1, 0]), torch.tensor([0, 1, 0, 0, 0]), (6. / 7., 0.66)),
        (torch.tensor([1, 1, 1, 1, 1]), torch.tensor([0, 0, 0, 0, 0]), (0, 0)),
        (torch.tensor([1, 0, 0, 0, 0]), torch.tensor([1, 0, 0, 0, 0]), (1, 1)),
        (torch.tensor([1, 1, 1, 0, 0]), torch.tensor([1, 1, 0, 0, 0]), (0.8, 0.8))
    ])
    def test_dice(self, prediction, label, expected, metrics):
        for idx, _ in enumerate(metrics.classes):
            if not np.isnan(expected[idx]):
                assert np.allclose(metrics._dice(prediction == idx, label == idx).item(), 
                                   np.array(expected[idx]), atol=1e-2)
            else:
                assert np.isnan(metrics._dice(prediction == idx, label == idx).item())