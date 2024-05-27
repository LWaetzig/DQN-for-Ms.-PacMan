import collections
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=self.capacity)

        self.size = 0

    def push(self, experience):
        self.buffer.append(experience)

        self.size += 1
        self.size = min(self.size, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.choice(self.size, batch_size)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )
        return (
            states,
            actions,
            rewards,
            dones,
            next_states,
        )
