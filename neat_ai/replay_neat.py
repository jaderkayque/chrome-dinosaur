# neat_ai/replay_neat.py
import pygame

pygame.init()

import os
import pickle
import sys

import numpy as np
import neat


from core.constants import *

from core.game import DinoEnv

from core.hud import HUD

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GENOME_PATH  = os.path.join(_ROOT, "neat_ai\models", "neat_best_official.pkl")
CONFIG_PATH = os.path.join(_ROOT, "neat_ai", "config.txt")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

ACTION_NAMES = {

    ACTION_RUN: "RUN",

    ACTION_JUMP: "JUMP",

    ACTION_DUCK: "DUCK",
}

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT → ACTION
# ══════════════════════════════════════════════════════════════════════════════

def outputs_to_action(outputs):

    return int(np.argmax(outputs))

# ══════════════════════════════════════════════════════════════════════════════
# LOAD NETWORK
# ══════════════════════════════════════════════════════════════════════════════

def load_network():

    if not os.path.exists(GENOME_PATH):

        raise FileNotFoundError(
            f"Genome not found: {GENOME_PATH}"
        )

    with open(GENOME_PATH, "rb") as f:

        genome = pickle.load(f)

    config = neat.Config(

        neat.DefaultGenome,

        neat.DefaultReproduction,

        neat.DefaultSpeciesSet,

        neat.DefaultStagnation,

        CONFIG_PATH
    )

    network = neat.nn.FeedForwardNetwork.create(

        genome,

        config
    )

    return network, genome

# ══════════════════════════════════════════════════════════════════════════════
# REPLAY
# ══════════════════════════════════════════════════════════════════════════════

def replay():

    network, genome = load_network()

    env = DinoEnv(
        render_mode=True
    )

    hud = HUD()

    print("\n=== NEAT REPLAY MODE ===\n")

    print(f"Loaded genome: {GENOME_PATH}")

    print("\nControls:")
    print("  ESC -> exit")
    print("  R   -> restart")
    print()

    # ══════════════════════════════════════════════════════════════════════════

    while True:

        state = env.reset()

        done = False

        total_reward = 0.0

        while not done:

            # ────────────────────────────────────────────────────────────────
            # EVENTS
            # ────────────────────────────────────────────────────────────────

            restart = False

            for event in pygame.event.get():

                if event.type == pygame.QUIT:

                    return

                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_ESCAPE:

                        return

                    if event.key == pygame.K_r:

                        restart = True

            if restart:
                break

            # ────────────────────────────────────────────────────────────────
            # NETWORK
            # ────────────────────────────────────────────────────────────────

            outputs = network.activate(state)

            action = outputs_to_action(outputs)

            next_state, reward, done, info = env.step(action)

            total_reward += reward

            state = next_state

            # ────────────────────────────────────────────────────────────────
            # HUD
            # ────────────────────────────────────────────────────────────────

            hud_data = {

                # generic
                "mode": "neat",

                "episode": 0,

                "score": env.points,

                "best_score": env.points,

                "speed": env.game_speed,

                "fps": env.clock.get_fps(),

                "reward": total_reward,

                "action_name":
                    ACTION_NAMES[action],

                # neat
                "alive": 1,

                "generation": 0,

                "fitness":
                    total_reward,

                "genome":
                    genome,

                "neural_outputs":
                    outputs,

                # sensors
                "state": state,

                "sensor_data":
                    info.get("sensor_data"),
            }

            env.render(
                hud=hud,
                hud_data=hud_data
            )

            env.tick(FPS)

        print(
            f"Game over | "
            f"Score: {env.points}"
        )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    replay()