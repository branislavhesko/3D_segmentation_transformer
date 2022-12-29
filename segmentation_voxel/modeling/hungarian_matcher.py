from scipy.optimize import linear_sum_assignment
import torch


def iou_cost(prediction, target):
    return (prediction.unsqueeze(1) == target.unsqueeze(2)).sum(dim=(3, 4)) / (prediction.unsqueeze(1) | target.unsqueeze(2)).sum(dim=(3, 4))


class HungarianMatcher(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, prediction, target):
        matches = []
        cost_matrix = iou_cost(prediction, target)
        for batch in range(cost_matrix.shape[0]):
            cm = 1 - cost_matrix[batch]
            rows, cols = linear_sum_assignment(cm.cpu().numpy())
            matches.append(torch.stack([torch.tensor(rows), torch.tensor(cols)], dim=1))
        return matches
