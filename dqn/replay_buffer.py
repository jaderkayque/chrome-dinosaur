# dqn/replay_buffer.py

import random
from collections import deque

import numpy as np
import torch

# ══════════════════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ══════════════════════════════════════════════════════════════════════════════
# REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:

    def __init__(self, capacity=100_000):

        self.buffer = deque(
            maxlen=capacity
        )

    # ─────────────────────────────────────────────────────────────────────────

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done,
            )
        )

    # ─────────────────────────────────────────────────────────────────────────

    def sample(self, batch_size):

        batch = random.sample(
            self.buffer,
            batch_size
        )

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = zip(*batch)

        return (

            torch.tensor(
                np.array(states),
                dtype=torch.float32
            ).to(DEVICE),

            torch.tensor(
                actions,
                dtype=torch.long
            ).to(DEVICE),

            torch.tensor(
                rewards,
                dtype=torch.float32
            ).to(DEVICE),

            torch.tensor(
                np.array(next_states),
                dtype=torch.float32
            ).to(DEVICE),

            torch.tensor(
                dones,
                dtype=torch.float32
            ).to(DEVICE),
        )

    # ─────────────────────────────────────────────────────────────────────────

    def clear(self):

        self.buffer.clear()

    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self):

        return len(self.buffer)