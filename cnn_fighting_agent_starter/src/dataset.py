import numpy as np
import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    """
    Expects an .npz with keys:
      - 'states':  (N, C_state, T, F_state)
      - 'buttons': (N, C_buttons, T, B)
      - 'labels':  (N,)
    """
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.states  = data['states'].astype(np.float32)
        self.buttons = data['buttons'].astype(np.float32)
        self.labels  = data['labels'].astype(np.int64)
        assert len(self.states) == len(self.buttons) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        s = torch.from_numpy(self.states[idx])    # (C_s, T, F_s)
        b = torch.from_numpy(self.buttons[idx])   # (C_b, T, B)
        y = torch.tensor(self.labels[idx])
        return (s, b), y