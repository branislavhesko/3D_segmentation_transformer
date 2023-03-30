import nibabel as nib
import numpy as np
import torch.utils.data as data
import torch


class AMOSLoader(data.Dataset):

    def __init__(self, volumetric_data, labels, target_shape) -> None:
        super().__init__()
        self.data = volumetric_data
        self.labels = labels
        self.target_shape = target_shape
        assert len(self.data) == len(self.labels)

    def __len__(self):
        return len(self.data)

    def _load_nii(self, path):
        img = nib.load(path)
        np_img = np.array(img.dataobj)
        return np_img / np.amax(np_img)

    def _align(self, data, target_shape):
        if data.shape == target_shape:
            return data
        elif any([data.shape[i] < target_shape[i] for i in range(3)]):
            pad = np.zeros(target_shape)
            pad[:data.shape[0], :data.shape[1], :data.shape[2]] = data
            return pad
        else:
            return data[:target_shape[0], :target_shape[1], :target_shape[2]]

    def __getitem__(self, item):
        sample = self.data[item]
        label = self.labels[item]
        sample = self._align(self._load_nii(sample), self.target_shape)
        label = self._align(self._load_nii(label), self.target_shape)
        label = torch.from_numpy(label).long()
        sample = torch.from_numpy(sample).float()
        return sample, label


def get_dataloader(cts, labels, target_shape, batch_size=1, shuffle=False, num_workers=0):
    dataset = AMOSLoader(cts, labels, target_shape)
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == "__main__":
    import glob
    import os

    path = "/Users/brani/DATA/amos/amos22"
    cts = sorted(glob.glob(os.path.join(path, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(path, "labelsTr", "*.nii.gz")))

    dataset = AMOSLoader(cts, labels, (64, 64, 64))

    for d in dataset:
        print(d[0].shape)
        print(d[1].shape)
        break