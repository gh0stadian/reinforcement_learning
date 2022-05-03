from collections import deque, namedtuple
from typing import Tuple
import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset


Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size, device):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))

        return (
            torch.from_numpy(np.array(states)).float().to(device),
            torch.from_numpy(np.array(actions)).long().to(device),
            torch.from_numpy(np.array(rewards, dtype=np.float32)).to(device),
            torch.from_numpy(np.array(dones, dtype=np.uint8)).to(device),
            torch.from_numpy(np.array(next_states)).float().to(device),
        )
