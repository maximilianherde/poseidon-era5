import torch
import h5py
from scOT.problems.base import BaseTimeDataset


class Advection(BaseTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 3600
        self.N_val = 120
        self.N_test = 150
        self.resolution = 128

        data_path = self.data_path + "/advected_blob_trajectories.h5"
        data_path = self._move_to_local_scratch(data_path)
        self.reader = h5py.File(data_path, "r")
        self.keys = list(self.reader.keys())

        self.constants = {
            "time": 20.0,
            "mean": torch.tensor([0.010812922, 3.05, 3.05]).unsqueeze(1).unsqueeze(1),
            "std": torch.tensor([0.07591591, 1.7318102, 1.7318102]).unsqueeze(1).unsqueeze(1),
        }

        self.input_dim = 3
        self.label_description = "[rho],[u,v]"

        self.post_init()

    def __getitem__(self, idx):
        i, t, t1, t2 = self._idx_map(idx)
        time = t / self.constants["time"]

        rho_input = torch.from_numpy(self.reader["data"][i + self.start, t1]).type(torch.float32)
        u_input = torch.Tensor([self.reader["velocity_values"][i + self.start][0]]).type(torch.float32).repeat(self.resolution, self.resolution)
        v_input = torch.Tensor([self.reader["velocity_values"][i + self.start][1]]).type(torch.float32).repeat(self.resolution, self.resolution)

        inputs = torch.stack([rho_input, u_input, v_input], dim=0)

        rho_label = torch.from_numpy(self.reader["data"][i + self.start, t2]).type(torch.float32)
        u_label = torch.Tensor([self.reader["velocity_values"][i + self.start][0]]).type(torch.float32).repeat(self.resolution, self.resolution)
        v_label = torch.Tensor([self.reader["velocity_values"][i + self.start][1]]).type(torch.float32).repeat(self.resolution, self.resolution)

        label = torch.stack([rho_label, u_label, v_label], dim=0)

        inputs = (inputs - self.constants["mean"]) / self.constants["std"]
        label = (label - self.constants["mean"]) / self.constants["std"]

        return {
            "pixel_values": inputs,
            "labels": label,
            "time": time,
        }
