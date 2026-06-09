# replay_dqn.py

import os
import sys

import pygame

pygame.init()

from core.constants import *

from core.game import DinoEnv

from core.hud import HUD

from dqn.agent import DQNAgent

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH  = os.path.join(_ROOT, "dqn\models", "dqn_best_official.pth")

ACTION_NAMES = {

    ACTION_RUN: "RUN",

    ACTION_JUMP: "JUMP",

    ACTION_DUCK: "DUCK",
}

# ══════════════════════════════════════════════════════════════════════════════
# REPLAY
# ══════════════════════════════════════════════════════════════════════════════

def replay(model_path=MODEL_PATH):

    if not os.path.exists(model_path):

        print(f"Model not found: {model_path}")

        return

    env = DinoEnv(
        render_mode=True
    )

    state_size = len(
        env.reset()
    )

    action_size = 3

    agent = DQNAgent(
        state_size,
        action_size
    )

    hud = HUD()

    agent.load(model_path)

    # no exploration
    agent.epsilon = 0.0

    print("\n=== DQN REPLAY MODE ===\n")

    print(f"Loaded model: {model_path}")

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
            # ACTION
            # ────────────────────────────────────────────────────────────────

            action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

            total_reward += reward

            state = next_state

            # ────────────────────────────────────────────────────────────────
            # HUD
            # ────────────────────────────────────────────────────────────────

            hud_data = {

                # generic
                "episode": 0,

                "score": env.points,

                "best_score": env.points,

                "speed": env.game_speed,

                "fps": env.clock.get_fps(),

                "reward": total_reward,

                "action_name":
                    ACTION_NAMES[action],

                # dqn
                "epsilon": 0.0,

                "loss": None,

                "q_values": agent.q_values,

                "memory_size": 0,

                "learn_steps": 0,

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

        # ────────────────────────────────────────────────────────────────────
        # DEATH SCREEN
        # ────────────────────────────────────────────────────────────────────

        print(
            f"Game over | "
            f"Score: {env.points}"
        )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    model_path = (

        sys.argv[1]

        if len(sys.argv) > 1

        else MODEL_PATH
    )

    replay(model_path)