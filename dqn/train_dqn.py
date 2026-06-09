# train_dqn.py

import pygame

pygame.init()

import os
import sys

import numpy as np
import torch

from core.constants import *

from core.game import DinoEnv

from core.hud import HUD

from dqn.agent import DQNAgent

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "dqn/models/dqn_latest.pth"

BEST_MODEL_PATH = "dqn/models/dqn_best.pth"

SAVE_EVERY = 25

MAX_EPISODES = 50_000

FAST_FPS = 0

NORMAL_FPS = FPS

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

ACTION_NAMES = {

    ACTION_RUN: "RUN",

    ACTION_JUMP: "JUMP",

    ACTION_DUCK: "DUCK",
}

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(resume=False):

    os.makedirs("models", exist_ok=True)

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

    if resume and os.path.exists(MODEL_PATH):

        agent.load(MODEL_PATH)

    episode = 0

    best_score = 0

    fast_mode = False

    step_count = 0

    print("\n=== DQN TRAINING STARTED ===\n")

    print("Controls:")
    print("  ESC -> save and exit")
    print("  F   -> toggle fast mode")
    print()

    # ══════════════════════════════════════════════════════════════════════════

    while episode < MAX_EPISODES:

        state = env.reset()

        done = False

        total_reward = 0.0

        last_action = ACTION_RUN

        while not done:

            # ────────────────────────────────────────────────────────────────
            # EVENTS
            # ────────────────────────────────────────────────────────────────

            for event in pygame.event.get():

                if event.type == pygame.QUIT:

                    agent.save(MODEL_PATH)

                    env.close()

                    return

                if event.type == pygame.KEYDOWN:

                    # exit
                    if event.key == pygame.K_ESCAPE:

                        agent.save(MODEL_PATH)

                        env.close()

                        return

                    # fast mode
                    if event.key == pygame.K_f:

                        fast_mode = not fast_mode

                        env.render_mode = not fast_mode

                        print(
                            f"Fast mode: "
                            f"{'ON' if fast_mode else 'OFF'}"
                        )

            # ────────────────────────────────────────────────────────────────
            # ACTION  +  FRAME SKIP
            # Agent picks one action; it is held for DQN_FRAME_SKIP frames.
            # Rewards accumulate; the last non-terminal state is stored.
            # ────────────────────────────────────────────────────────────────

            action = agent.select_action(state)

            last_action = action

            accumulated_reward = 0.0
            next_state = state
            info = {}

            for _ in range(DQN_FRAME_SKIP):

                if done:
                    break

                next_state, r, done, info = env.step(action)

                accumulated_reward += r

                # render every game frame for smooth visuals
                if env.render_mode:

                    hud_data = {
                        "episode":      episode + 1,
                        "score":        env.points,
                        "best_score":   best_score,
                        "speed":        env.game_speed,
                        "fps":          env.clock.get_fps(),
                        "reward":       total_reward,
                        "action_name":  ACTION_NAMES[action],
                        "epsilon":      agent.epsilon,
                        "loss":         agent.last_loss,
                        "q_values":     agent.q_values,
                        "memory_size":  len(agent.memory),
                        "learn_steps":  agent.learn_steps,
                        "steps":        step_count,
                        "state":        next_state,
                        "sensor_data":  info.get("sensor_data"),
                    }

                    env.render(hud=hud, hud_data=hud_data)

                    env.tick(FAST_FPS if fast_mode else NORMAL_FPS)

            total_reward += accumulated_reward

            # ────────────────────────────────────────────────────────────────
            # MEMORY
            # ────────────────────────────────────────────────────────────────

            agent.remember(state, action, accumulated_reward, next_state, float(done))

            state = next_state

            step_count += 1

            # ────────────────────────────────────────────────────────────────
            # TRAIN STEP
            # ────────────────────────────────────────────────────────────────

            loss = agent.train_step() if step_count % DQN_TRAIN_EVERY == 0 else None

            if loss is not None:
                hud.push_loss(loss)

        # ══════════════════════════════════════════════════════════════════
        # END EPISODE
        # ══════════════════════════════════════════════════════════════════

        episode += 1

        agent.decay_epsilon()

        hud.push_episode(
            env.points,
            total_reward
        )

        # save best
        if env.points > best_score:

            best_score = env.points

            agent.save(BEST_MODEL_PATH)

        # autosave
        if episode % SAVE_EVERY == 0:

            agent.save(MODEL_PATH)

        # console log
        loss_str = (
            f"{agent.last_loss:.5f}"
            if agent.last_loss is not None
            else "-"
        )

        print(

            f"EP {episode:5d} | "

            f"Score: {env.points:5d} | "

            f"Best: {best_score:5d} | "

            f"ε: {agent.epsilon:.4f} | "

            f"Reward: {total_reward:8.2f} | "

            f"Loss: {loss_str}"
        )

    # ══════════════════════════════════════════════════════════════════════════

    agent.save(MODEL_PATH)

    env.close()

    print("\nTraining finished.\n")

# ══════════════════════════════════════════════════════════════════════════════
# REPLAY
# ══════════════════════════════════════════════════════════════════════════════

def replay(model_path=BEST_MODEL_PATH):

    if not os.path.exists(model_path):

        print(f"Model not found: {model_path}")

        return

    env = DinoEnv(
        render_mode=True
    )

    state_size = len(
        env.reset()
    )

    agent = DQNAgent(
        state_size,
        3
    )

    hud = HUD()

    agent.load(model_path)

    agent.epsilon = 0.0

    print(f"\nReplay mode: {model_path}\n")

    while True:

        state = env.reset()

        done = False

        total_reward = 0.0

        while not done:

            for event in pygame.event.get():

                if event.type == pygame.QUIT:

                    pygame.quit()

                    return

                if (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_ESCAPE
                ):

                    pygame.quit()

                    return

            action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

            total_reward += reward

            state = next_state

            hud_data = {

                "episode": 0,

                "score": env.points,

                "best_score": env.points,

                "speed": env.game_speed,

                "fps": env.clock.get_fps(),

                "reward": total_reward,

                "action_name":
                    ACTION_NAMES[action],

                "epsilon": 0.0,

                "loss": None,

                "q_values": agent.q_values,

                "memory_size": 0,

                "learn_steps": 0,

                "state": state,

                "sensor_data":
                    info.get("sensor_data"),
            }

            env.render(
                hud=hud,
                hud_data=hud_data
            )

            env.tick(FPS)

        print(f"Replay score: {env.points}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    mode = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "--train"
    )

    if mode == "--resume":

        print("Resuming DQN training...\n")

        train(resume=True)

    elif mode == "--replay":

        replay()

    else:

        print("Starting DQN training...\n")

        train(resume=False)