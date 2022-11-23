import numpy as np


class Metrics:
    
    def __init__(self, classes) -> None:
        self.iou = {class_: [] for class_ in classes}
        self.dice = {class_: [] for class_ in classes}
        self.classes = {class_: idx for idx, class_ in enumerate(classes)}
        
    def update(self, prediction, label):
        for class_, idx in self.classes.items():
            iou = self._iou(prediction == idx, label == idx)
            dice = self._dice(prediction == idx, label == idx)
            self.iou[class_].append(float(iou.item()))
            self.dice[class_].append(float(dice.item()))
        
    def _iou(self, prediction, label):
        return (prediction & label).sum().float() / (
            prediction.sum().float() + label.sum().float() - (prediction & label).sum().float())
        
    def _dice(self, prediction, label):
        return 2 * (prediction & label).sum().float() / (
            prediction.sum().float() + label.sum().float())

    def __str__(self) -> str:
        out = ""
        for class_, idx in self.classes.items():
            out += f"IOU_{class_}: {np.mean(self.iou[class_])}" + "\n"
            out += f"DICE_{class_}: {np.mean(self.dice[class_])}" "\n"
        return out