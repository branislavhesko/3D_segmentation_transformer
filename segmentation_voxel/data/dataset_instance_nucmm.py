from segmentation_voxel.data.dataset_nucmm import NUCMMLoader


class NUCMMInstanceLoader(NUCMMLoader):

    def _get_label(self, label):
        return label
