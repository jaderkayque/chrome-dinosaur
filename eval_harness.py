# eval_harness.py — measures REAL game scores for trained D3QN and NEAT agents.
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
pygame.init()

import sys
sys.path.insert(0, ".")

import random
import pickle
import numpy as np

from core.game import DinoEnv
from core.constants import N_OBS, N_ACTIONS, ACTION_RUN
import neat
from dqn.agent import DQNAgent

N_EPISODES = 50
MAX_FRAMES = 20_000  # hard cap so an immortal policy still terminates an episode

DQN_PATH  = "dqn/models/dqn_best.pth"
NEAT_PATH = "neat_ai/models/neat_best.pkl"
NEAT_CFG  = "neat_ai/config.txt"


def run_dqn(env, agent, seed):
    random.seed(seed)
    np.random.seed(seed)
    state = env.reset()
    done = False
    frames = 0
    while not done and frames < MAX_FRAMES:
        a = agent.select_action(state)
        state, _, done, _ = env.step(a)
        frames += 1
    return env.points


def run_neat(env, net, seed):
    random.seed(seed)
    np.random.seed(seed)
    state = env.reset()
    done = False
    frames = 0
    while not done and frames < MAX_FRAMES:
        out = net.activate(state)
        a = int(np.argmax(out))
        state, _, done, _ = env.step(a)
        frames += 1
    return env.points


def summarize(name, scores):
    s = np.array(scores, dtype=np.float64)
    print(f"\n=== {name} (n={len(s)}) ===")
    print(f"  mean   = {s.mean():.1f}")
    print(f"  std    = {s.std(ddof=1):.1f}")
    print(f"  median = {np.median(s):.1f}")
    print(f"  min    = {s.min():.0f}")
    print(f"  max    = {s.max():.0f}")
    print(f"  q25    = {np.percentile(s,25):.0f}")
    print(f"  q75    = {np.percentile(s,75):.0f}")
    return s


def main():
    env = DinoEnv(render_mode=False)

    # DQN
    agent = DQNAgent(N_OBS, N_ACTIONS)
    agent.load(DQN_PATH)
    agent.epsilon = 0.0
    dqn_scores = [run_dqn(env, agent, seed=1000 + i) for i in range(N_EPISODES)]
    summarize("D3QN", dqn_scores)

    # NEAT
    with open(NEAT_PATH, "rb") as f:
        genome = pickle.load(f)
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation, NEAT_CFG)
    net = neat.nn.FeedForwardNetwork.create(genome, cfg)
    neat_scores = [run_neat(env, net, seed=1000 + i) for i in range(N_EPISODES)]
    summarize("NEAT", neat_scores)

    # identical-seed paired comparison (same obstacle sequences)
    np.save("eval_dqn_scores.npy", np.array(dqn_scores))
    np.save("eval_neat_scores.npy", np.array(neat_scores))
    print("\nSaved eval_dqn_scores.npy / eval_neat_scores.npy")


if __name__ == "__main__":
    main()
