import torch
import h5py
from scOT.problems.base import BaseTimeDataset


class Triple(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 5014 // (self.time_step_size * self.max_num_time_steps)
        self.N_val = 1
        self.N_test = 5
        self.resolution = 128

        data_path = self.data_path + "/triple.h5"
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")

        self.constants = {
            "mean": torch.tensor(
                [
                    0.709753493154533,
                    0.01810408939432013,
                    0.007216560674947469,
                    0.37139811680921125,
                ]
            ),
            "std": torch.tensor(
                [
                    0.5340755544511663,
                    0.2691626641198381,
                    0.16361558298495668,
                    0.16093527025207108,
                ]
            ),
            "time": 25.0,
        }

        self.input_dim = 4
        self.label_description = "[rho],[u,v],[p]"

        self.pixel_mask = torch.tensor([False, False, False, False])

        self.post_init()

    def __getitem__(self, idx):
        i, _, t1, t2 = self._idx_map(idx)
        time = self.reader["time"][t2] - self.reader["time"][t1]
        time = time / self.constants["time"]

        inputs = (
            torch.from_numpy(self.reader["data"][i + self.start + t1, 0:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        label = (
            torch.from_numpy(self.reader["data"][i + self.start + t2, 0:4])
            .type(torch.float32)
            .reshape(4, self.resolution, self.resolution)
        )
        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        label = (label - self.constants["mean"]) / self.constants["std"]

        return {
            "pixel_values": inputs,
            "labels": label,
            "time": time,
            "pixel_mask": self.pixel_mask,
        }
