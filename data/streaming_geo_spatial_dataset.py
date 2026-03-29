import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioError
from torch.utils.data.dataset import IterableDataset
from torch.utils.data.dataset import Dataset
from rasterio.errors import RasterioIOError


class StreamingGeospatialDataset(IterableDataset):
    def __init__(
        self,
        imagery_fns,
        label_fns=None,
        groups=None,
        chip_size=256,
        num_chips_per_tile=200,
        windowed_sampling=False,
        image_transform=None,
        label_transform=None,
        nodata_check=None,
        verbose=False,
    ):
        if label_fns is None:
            self.fns = imagery_fns
            self.use_labels = False
        else:
            self.fns = list(zip(imagery_fns, label_fns))

            self.use_labels = True

        self.groups = groups

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile
        self.windowed_sampling = windowed_sampling

        self.image_transform = image_transform
        self.label_transform = label_transform
        self.nodata_check = nodata_check

        self.verbose = verbose

        if self.verbose:
            print("Constructed StreamingGeospatialDataset")

    def stream_tile_fns(self):
        worker_info = torch.utils.data.get_worker_info()
        if (
            worker_info is None
        ):  # In this case we are not loading through a DataLoader with multiple workers
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        # We only want to shuffle the order we traverse the files if we are the first worker (else, every worker will shuffle the files...)
        if worker_id == 0:
            np.random.shuffle(self.fns)  # in place

        if self.verbose:
            print("Creating a filename stream for worker %d" % (worker_id))

        # This logic splits up the list of filenames into `num_workers` chunks. Each worker will recieve ceil(num_filenames / num_workers) filenames to generate chips from. If the number of workers doesn't divide the number of filenames evenly then the last worker will have fewer filenames.
        N = len(self.fns)
        num_files_per_worker = int(np.ceil(N / num_workers))
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(N, (worker_id + 1) * num_files_per_worker)
        for idx in range(lower_idx, upper_idx):
            label_fn = None
            if self.use_labels:
                img_fn, label_fn = self.fns[idx]
            else:
                img_fn = self.fns[idx]

            if self.groups is not None:
                group = self.groups[idx]
            else:
                group = None

            if self.verbose:
                print("Worker %d, yielding file %d" % (worker_id, idx))

            yield (img_fn, label_fn, group)

    def stream_chips(self):
        for img_fn, label_fn, group in self.stream_tile_fns():
            num_skipped_chips = 0

            # Open file pointers
            img_fp = rasterio.open(img_fn, "r")
            label_fp = rasterio.open(label_fn, "r") if self.use_labels else None

            height, width = img_fp.shape
            if (
                self.use_labels
            ):  # garuntee that our label mask has the same dimensions as our imagery
                t_height, t_width = label_fp.shape
                assert height == t_height and width == t_width

            # If we aren't in windowed sampling mode then we should read the entire tile up front
            img_data = None
            label_data = None
            try:
                if not self.windowed_sampling:
                    img_data = np.rollaxis(img_fp.read(3), 0, 3)
                    if self.use_labels:
                        label_data = (
                            label_fp.read().squeeze()
                        )  # assume the label geotiff has a single channel
            except RasterioError as e:
                print(
                    "WARNING: Error reading in entire file, skipping to the next file"
                )
                continue

            for i in range(self.num_chips_per_tile):
                # Select the top left pixel of our chip randomly
                x = np.random.randint(0, width - self.chip_size)
                y = np.random.randint(0, height - self.chip_size)

                # Read imagery / labels
                img = None
                labels = None
                if self.windowed_sampling:
                    try:
                        img = np.rollaxis(
                            img_fp.read(
                                window=Window(x, y, self.chip_size, self.chip_size)
                            ),
                            0,
                            3,
                        )
                        # print(img.shape)
                        if self.use_labels:
                            labels = label_fp.read(
                                window=Window(x, y, self.chip_size, self.chip_size)
                            ).squeeze()
                    except RasterioError:
                        print(
                            "WARNING: Error reading chip from file, skipping to the next chip"
                        )
                        continue
                else:
                    img = img_data[y : y + self.chip_size, x : x + self.chip_size, :]
                    if self.use_labels:
                        labels = label_data[
                            y : y + self.chip_size, x : x + self.chip_size
                        ]

                # Check for no data
                if self.nodata_check is not None:
                    if self.use_labels:
                        skip_chip = self.nodata_check(img, labels)
                    else:
                        skip_chip = self.nodata_check(img)

                    if (
                        skip_chip
                    ):  # The current chip has been identified as invalid by the `nodata_check(...)` method
                        num_skipped_chips += 1
                        continue

                # Transform the imagery
                if self.image_transform is not None:
                    if self.groups is None:
                        img = self.image_transform(img)
                    else:
                        img = self.image_transform(img, group)
                else:
                    img = torch.from_numpy(img).squeeze()

                # Transform the labels
                if self.use_labels:
                    if self.label_transform is not None:
                        if self.groups is None:
                            labels = self.label_transform(labels)
                        else:
                            print(label_fn)
                            labels = self.label_transform(labels, group)
                            print(labels)
                    else:
                        labels = torch.from_numpy(labels).squeeze()

                # Note, that img should be a torch "Double" type (i.e. a np.float32) and labels should be a torch "Long" type (i.e. np.int64)
                if self.use_labels:
                    yield img, labels
                else:
                    yield img
            # Close file pointers
            img_fp.close()
            if self.use_labels:
                label_fp.close()

            if num_skipped_chips > 0 and self.verbose:
                print("We skipped %d chips on %s" % (img_fn))

    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamingGeospatialDataset iterator")
        return iter(self.stream_chips())


class TileInferenceDataset(Dataset):
    def __init__(
        self,
        fn,
        chip_size,
        stride,
        gt=None,
        transform=None,
        label_transform=None,
        windowed_sampling=False,
        verbose=False,
    ):
        self.fn = fn
        self.gt_fn = gt
        self.chip_size = chip_size

        self.transform = transform
        self.label_transform = label_transform
        self.windowed_sampling = windowed_sampling
        self.verbose = verbose

        with rasterio.open(self.fn) as f:
            height, width = f.height, f.width
            self.num_channels = f.count
            self.dtype = f.profile["dtype"]
            if (
                not windowed_sampling
            ):  # if we aren't using windowed sampling, then go ahead and read in all of the data
                self.data = np.rollaxis(f.read(), 0, 3)

        self.gt_data = None
        self.gt_num_channels = None
        self.gt_dtype = None
        if self.gt_fn is not None:
            with rasterio.open(self.gt_fn) as f:
                gt_height, gt_width = f.height, f.width
                self.gt_num_channels = f.count
                self.gt_dtype = f.profile["dtype"]
                if not windowed_sampling:
                    self.gt_data = np.rollaxis(f.read(), 0, 3)
                # Assume dimensions match the input for simplicity
                if (gt_height, gt_width) != (height, width):
                    raise ValueError("Ground truth dimensions do not match input dimensions.")

        self.chip_coordinates = (
            []
        )  # upper left coordinate (y,x), of each chip that this Dataset will return
        for y in list(range(0, height - self.chip_size, stride)) + [
            height - self.chip_size
        ]:
            for x in list(range(0, width - self.chip_size, stride)) + [
                width - self.chip_size
            ]:
                self.chip_coordinates.append((y, x))
        self.num_chips = len(self.chip_coordinates)

        if self.verbose:
            print(
                "Constructed TileInferenceDataset -- we have %d by %d file with %d channels with a dtype of %s. We are sampling %d chips from it."
                % (height, width, self.num_channels, self.dtype, self.num_chips)
            )
            if self.gt_fn is not None:
                print(
                    "Ground truth: %d by %d file with %d channels with a dtype of %s."
                    % (height, width, self.gt_num_channels, self.gt_dtype)
                )

    def __getitem__(self, idx):
        y, x = self.chip_coordinates[idx]

        if self.windowed_sampling:
            try:
                with rasterio.Env():
                    with rasterio.open(self.fn) as f:
                        img = np.rollaxis(
                            f.read(
                                window=rasterio.windows.Window(
                                    x, y, self.chip_size, self.chip_size
                                )
                            ),
                            0,
                            3,
                        )
            except (
                RasterioIOError
            ) as e:  # NOTE(caleb): I put this here to catch weird errors that I was seeing occasionally when trying to read from COGS - I don't remember the details though
                print("Reading %d failed, returning 0's" % (idx))
                img = np.zeros(
                    (self.chip_size, self.chip_size, self.num_channels), dtype=np.uint8
                )
        else:
            img = self.data[y : y + self.chip_size, x : x + self.chip_size]

        gt_img = None
        if self.gt_fn is not None:
            if self.windowed_sampling:
                try:
                    with rasterio.Env():
                        with rasterio.open(self.gt_fn) as f:
                            gt_img = np.rollaxis(
                                f.read(
                                    window=rasterio.windows.Window(
                                        x, y, self.chip_size, self.chip_size
                                    )
                                ),
                                0,
                                3,
                            )
                except RasterioIOError as e:
                    print("Reading GT %d failed, returning 0's" % (idx))
                    gt_img = np.zeros(
                        (self.chip_size, self.chip_size, self.gt_num_channels), dtype=np.uint8
                    )
            else:
                gt_img = self.gt_data[y : y + self.chip_size, x : x + self.chip_size]

        if self.transform is not None:
            img = self.transform(img)
        if gt_img is not None and self.label_transform is not None:
            gt_img = self.label_transform(gt_img)

        if gt_img is not None:
            return img, gt_img, np.array((y, x))
        else:
            return img, np.array((y, x))

    def __len__(self):
        return self.num_chips
