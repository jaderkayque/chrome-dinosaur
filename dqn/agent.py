# dqn/agent.py

import os
import random
import tempfile
import time

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from dqn.model import DuelingDQN

# ══════════════════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ══════════════════════════════════════════════════════════════════════════════
# SUM TREE  (O(log n) sampling for PER)
# ══════════════════════════════════════════════════════════════════════════════

class SumTree:

    def __init__(self, capacity):

        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data      = np.empty(capacity, dtype=object)
        self.write     = 0
        self.n_entries = 0

    # ─────────────────────────────────────────────────────────────────────────

    def _propagate(self, idx, delta):

        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    # ─────────────────────────────────────────────────────────────────────────

    def _retrieve(self, idx, s):

        while True:
            left  = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s  -= self.tree[left]
                idx = right

    # ─────────────────────────────────────────────────────────────────────────

    def total(self):
        return float(self.tree[0])

    # ─────────────────────────────────────────────────────────────────────────

    def add(self, priority, data):

        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.capacity

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # ─────────────────────────────────────────────────────────────────────────

    def update(self, idx, priority):

        delta          = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    # ─────────────────────────────────────────────────────────────────────────

    def get(self, s):

        idx      = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]

    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self):
        return self.n_entries


# ══════════════════════════════════════════════════════════════════════════════
# PRIORITIZED REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class PrioritizedReplayBuffer:

    def __init__(self, capacity, alpha=0.6):

        self.alpha        = alpha
        self.tree         = SumTree(capacity)
        self.max_priority = 1.0

    # ─────────────────────────────────────────────────────────────────────────

    def push(self, state, action, reward, next_state, done):

        self.tree.add(
            self.max_priority ** self.alpha,
            (state, action, reward, next_state, done)
        )

    # ─────────────────────────────────────────────────────────────────────────

    def sample(self, batch_size, beta):

        total    = self.tree.total()
        segment  = total / batch_size

        indices    = []
        priorities = []
        samples    = []

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            # guard against empty slots
            if data is None:
                s    = random.uniform(0, total)
                idx, p, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(max(p, 1e-8))
            samples.append(data)

        # importance-sampling weights
        probs   = np.array(priorities) / total
        weights = (probs * self.tree.n_entries) ** (-beta)
        weights /= weights.max()          # normalize so max weight == 1
        weights  = weights.astype(np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            torch.tensor(np.array(states),      dtype=torch.float32).to(DEVICE),
            torch.tensor(actions,               dtype=torch.long   ).to(DEVICE),
            torch.tensor(rewards,               dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE),
            torch.tensor(dones,                 dtype=torch.float32).to(DEVICE),
            indices,
            torch.tensor(weights,               dtype=torch.float32).to(DEVICE),
        )

    # ─────────────────────────────────────────────────────────────────────────

    def update_priorities(self, indices, td_errors):

        for idx, err in zip(indices, td_errors):
            p = (float(abs(err)) + 1e-6) ** self.alpha
            self.tree.update(idx, p)
            if p > self.max_priority:
                self.max_priority = p

    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.tree)


# ══════════════════════════════════════════════════════════════════════════════
# DQN AGENT
# ══════════════════════════════════════════════════════════════════════════════

class DQNAgent:

    GAMMA = 0.99

    LEARNING_RATE = 1e-3

    BATCH_SIZE = 128

    BUFFER_SIZE = 100_000

    MIN_REPLAY_SIZE = 2_000

    # soft target update coefficient (applied every learn step)
    TAU = 0.005

    EPSILON_START = 1.0
    EPSILON_END   = 0.02
    EPSILON_DECAY = 0.995

    # PER
    PER_ALPHA         = 0.6
    PER_BETA_START    = 0.4
    PER_BETA_INCREMENT = 2e-7   # reaches ~1.0 after ~3M learn steps

    # ══════════════════════════════════════════════════════════════════════════

    def __init__(
        self,
        state_size,
        action_size
    ):

        self.state_size  = state_size
        self.action_size = action_size

        self.epsilon     = self.EPSILON_START
        self.beta        = self.PER_BETA_START
        self.learn_steps = 0

        # networks
        self.policy_net = DuelingDQN(state_size, action_size).to(DEVICE)
        self.target_net = DuelingDQN(state_size, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.LEARNING_RATE
        )

        # prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(
            self.BUFFER_SIZE,
            alpha=self.PER_ALPHA
        )

        # metrics
        self.last_loss = None
        self.q_values  = None

    # ══════════════════════════════════════════════════════════════════════════
    # ACTION SELECTION
    # ══════════════════════════════════════════════════════════════════════════

    def select_action(self, state):

        with torch.no_grad():

            state_t = torch.tensor(
                state, dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)

            q = self.policy_net(state_t)

            self.q_values = (
                q.squeeze().detach().cpu().numpy()
            )

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        return int(q.argmax(dim=1).item())

    # ══════════════════════════════════════════════════════════════════════════
    # MEMORY
    # ══════════════════════════════════════════════════════════════════════════

    def remember(self, state, action, reward, next_state, done):

        self.memory.push(state, action, reward, next_state, float(done))

    # ══════════════════════════════════════════════════════════════════════════
    # TRAINING
    # ══════════════════════════════════════════════════════════════════════════

    def train_step(self):

        if len(self.memory) < self.MIN_REPLAY_SIZE:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            is_weights,
        ) = self.memory.sample(self.BATCH_SIZE, self.beta)

        # anneal beta toward 1.0
        self.beta = min(1.0, self.beta + self.PER_BETA_INCREMENT)

        # current Q values for taken actions
        current_q = self.policy_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # double DQN target
        with torch.no_grad():

            next_actions = self.policy_net(next_states).argmax(dim=1)

            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

            target_q = rewards + self.GAMMA * next_q * (1 - dones)

        # per-sample TD errors → update priorities
        td_errors = (target_q - current_q).detach().abs().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        # IS-weighted Huber loss
        elementwise_loss = F.smooth_l1_loss(
            current_q, target_q, reduction="none"
        )
        loss = (is_weights * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # soft target update (every learn step)
        for p_param, t_param in zip(
            self.policy_net.parameters(),
            self.target_net.parameters()
        ):
            t_param.data.copy_(
                self.TAU * p_param.data + (1 - self.TAU) * t_param.data
            )

        self.last_loss    = loss.item()
        self.learn_steps += 1

        return self.last_loss

    # ══════════════════════════════════════════════════════════════════════════
    # EPSILON
    # ══════════════════════════════════════════════════════════════════════════

    def decay_epsilon(self):

        self.epsilon = max(
            self.EPSILON_END,
            self.epsilon * self.EPSILON_DECAY
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SAVE / LOAD
    # ══════════════════════════════════════════════════════════════════════════

    def save(self, path):

        payload = {
            "policy_net":  self.policy_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "epsilon":     self.epsilon,
            "beta":        self.beta,
            "learn_steps": self.learn_steps,
        }

        # Write to a temp file first, then rename — avoids OneDrive/Defender
        # locking the existing .pth during sync (Windows error 32).
        dir_ = os.path.dirname(os.path.abspath(path))
        os.makedirs(dir_, exist_ok=True)

        for attempt in range(4):
            try:
                fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
                os.close(fd)
                torch.save(payload, tmp)
                os.replace(tmp, path)
                print(f"[DQN] Model saved: {path}")
                return
            except (RuntimeError, PermissionError, OSError):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                if attempt < 3:
                    time.sleep(1)

        print(f"[DQN] WARNING: could not save model to {path}")

    # ─────────────────────────────────────────────────────────────────────────

    def load(self, path):

        checkpoint = torch.load(path, map_location=DEVICE)

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.epsilon     = checkpoint.get("epsilon",     self.EPSILON_END)
        self.beta        = checkpoint.get("beta",        self.PER_BETA_START)
        self.learn_steps = checkpoint.get("learn_steps", 0)

        print(f"[DQN] Model loaded: {path}")
