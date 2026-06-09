# dqn/model.py

import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════════════════════
# DUELING DQN
# ══════════════════════════════════════════════════════════════════════════════

class DuelingDQN(nn.Module):

    def __init__(
        self,
        n_inputs,
        n_actions,
        hidden_size=256
    ):

        super().__init__()

        # feature extractor
        self.features = nn.Sequential(

            nn.Linear(n_inputs, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # value stream
        self.value_stream = nn.Sequential(

            nn.Linear(hidden_size, 128),
            nn.ReLU(),

            nn.Linear(128, 1),
        )

        # advantage stream
        self.advantage_stream = nn.Sequential(

            nn.Linear(hidden_size, 128),
            nn.ReLU(),

            nn.Linear(128, n_actions),
        )

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x):

        features = self.features(x)

        value = self.value_stream(features)

        advantage = self.advantage_stream(features)

        q_values = value + (
            advantage - advantage.mean(dim=1, keepdim=True)
        )

        return q_values