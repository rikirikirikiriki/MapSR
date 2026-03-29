from .streaming_geo_spatial_dataset import StreamingGeospatialDataset
from .transforms import image_transforms, label_transforms
import pandas as pd
import numpy as np


def nodata_check(img, labels):
    return np.any(labels == 0) or np.any(np.sum(img == 0, axis=2) == 4)


class TrainDataset(StreamingGeospatialDataset):
    def __init__(self, list_dir, chip_size=224, num_chips_per_tile=50):
        """
        list_dir: Path to the CSV file containing image and label file paths
        chip_size: Size of each sampled chip
        num_chips_per_tile: How many chips will be sampled from one large-scale tile
        """
        input_dataframe = pd.read_csv(list_dir)
        self.image_fns = input_dataframe["image_fn"].values
        self.label_fns = input_dataframe["label_fn"].values

        super(TrainDataset, self).__init__(
            imagery_fns=self.image_fns,
            label_fns=self.label_fns,
            groups=None,
            chip_size=chip_size,
            num_chips_per_tile=num_chips_per_tile,
            windowed_sampling=True,
            verbose=False,
            image_transform=image_transforms,
            label_transform=label_transforms,
            nodata_check=nodata_check,
        )
