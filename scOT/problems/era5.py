import torch
import h5py
from scOT.problems.base import BaseTimeDataset


class ERA5_UV(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = (2010 - 1995) * 365 * 4 - self.max_num_time_steps + 1
        self.N_val = (2007 - 2005) * 365 * 4 - self.max_num_time_steps + 1
        self.N_test = (2010 - 2007) * 365 * 4 - self.max_num_time_steps + 1
        self.resolution = 128

        data_path = self.data_path + "/ERA5.h5"
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")
        self.keys = list(self.reader.keys())

        self.constants = {"time": self.max_num_time_steps,
                          "mean": torch.tensor([-0.05483689, 0.18707459]).unsqueeze(1).unsqueeze(1),
                          "std": torch.tensor([5.2594, 4.5301833]).unsqueeze(1).unsqueeze(1)}

        self.input_dim = 2
        self.label_description = "[10U,10V]"

        self.post_init()

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]

        input_tensors = list(torch.from_numpy(
                self.reader[self.keys[idx]][i+self.start + t1][:, :-1]).type(torch.float32) for idx in range(len(self.keys)))

        inputs = torch.stack(input_tensors, dim=0)

        label_tensors = list(torch.from_numpy(
                self.reader[self.keys[idx]][i+self.start + t2][:, :-1]).type(torch.float32) for idx in range(len(self.keys)))


        label = torch.stack(label_tensors, dim=0)

        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        label = (label - self.constants["mean"]) / self.constants["std"]

        return {
            "pixel_values": inputs,
            "labels": label,
            "time": time,
        }
