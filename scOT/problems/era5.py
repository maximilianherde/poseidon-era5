import torch
import h5py
from scOT.problems.base import BaseTimeDataset


class ERA5_UV(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = (2022 - 1995) * 365 * 4 + 7 * 4 - self.max_num_time_steps + 1
        self.N_val = (2015 - 2012) * 365 * 4 + 1 * 4 - self.max_num_time_steps + 1
        self.N_test = (2022 - 2015) * 365 * 4 + 2 * 4 - self.max_num_time_steps + 1
        self.resolution = 128

        data_path = self.data_path + "/ERA5.h5"
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = {"time": self.max_num_time_steps}  # TODO

        self.input_dim = 2
        self.label_description = "[10U,10V]"

        self.post_init()

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]

        inputs = None
        labels = None

        return {
            "inputs": inputs,
            "labels": labels,
            "time": time,
        }
