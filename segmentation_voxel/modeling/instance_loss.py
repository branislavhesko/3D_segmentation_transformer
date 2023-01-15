import torch.nn as nn

from segmentation_voxel.modeling.hungarian_matcher import HungarianMatcher


class InstanceLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.loss = nn.BCELoss()
        self.matcher = HungarianMatcher()

    def forward(self, prediction, target):
        # prediction: [batch_size, num_classes, height, width]
        # target: [batch_size, height, width]
        matches = self.matcher(prediction > 0.5, target)
        loss = 0
        for batch, match in enumerate(matches):
            for row, col, cost in match:
                if cost < 0.5:
                    continue
                loss += self.loss(prediction[batch, int(col), ...], target[batch, int(row), ...].float())
        return self.weight * loss / len(matches)


if __name__ == "__main__":
    import torch

    prediction = torch.rand(2, 3, 4, 5)
    target = torch.rand(2, 5, 4, 5) > 0.5

    criterion = InstanceLoss()
    loss = criterion(prediction, target)
    print(loss)
