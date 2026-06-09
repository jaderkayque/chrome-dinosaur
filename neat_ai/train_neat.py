# neat_ai/train_neat.py
import pygame

pygame.init()

import glob
import os
import pickle
import sys

import numpy as np
import neat

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING ABORTED SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

class TrainingAborted(Exception):
    pass


from core.constants import *

from core.game import DinoEnv

from core.hud import HUD

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

CONFIG_PATH = "neat_ai/config.txt"

BEST_GENOME_PATH = "neat_ai/models/neat_best.pkl"

MAX_GENERATIONS = 300

# ══════════════════════════════════════════════════════════════════════════════
# GLOBALS
# ══════════════════════════════════════════════════════════════════════════════

generation = 0

best_score = 0

best_genome = None

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
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def eval_genomes(genomes, config):

    global generation, best_score, best_genome

    generation += 1

    env = DinoEnv(render_mode=True)
    hud = HUD()

    n = len(genomes)

    nets         = []
    genome_list  = []

    for _, genome in genomes:
        genome.fitness = 0.0
        nets.append(
            neat.nn.FeedForwardNetwork.create(genome, config)
        )
        genome_list.append(genome)

    env.reset(n_dinos=n)
    states = [env._get_obs_for(d) for d in env.dinos]

    # ══════════════════════════════════════════════════════════════════════════

    while not env.done:

        # ────────────────────────────────────────────────────────────────────
        # EVENTS
        # ────────────────────────────────────────────────────────────────────

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                if best_genome is not None:
                    save_best_genome(best_genome)
                raise TrainingAborted()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if best_genome is not None:
                        save_best_genome(best_genome)
                    raise TrainingAborted()

        # ────────────────────────────────────────────────────────────────────
        # ACTIONS — all alive dinos act in parallel
        # ────────────────────────────────────────────────────────────────────

        actions     = []
        outputs_all = []

        for i in range(n):
            if env.alive[i]:
                out = nets[i].activate(states[i])
                outputs_all.append(out)
                actions.append(outputs_to_action(out))
            else:
                outputs_all.append([0.0, 0.0, 0.0])
                actions.append(ACTION_RUN)

        # ────────────────────────────────────────────────────────────────────
        # STEP
        # ────────────────────────────────────────────────────────────────────

        obs_list, rewards, dones, info = env.step_multi(actions)

        for i in range(n):
            genome_list[i].fitness += rewards[i]
            if obs_list[i] is not None:
                states[i] = obs_list[i]

        # ────────────────────────────────────────────────────────────────────
        # TRACK BEST SCORE (save happens once, after generation ends)
        # ────────────────────────────────────────────────────────────────────

        if env.points > best_score:
            best_score = env.points

        # ────────────────────────────────────────────────────────────────────
        # HUD — show stats for the best alive genome
        # ────────────────────────────────────────────────────────────────────

        alive_indices = [j for j in range(n) if env.alive[j]]

        hud_i = (
            max(alive_indices, key=lambda j: genome_list[j].fitness)
            if alive_indices else 0
        )

        hud_data = {

            "mode": "neat",

            "episode":    generation,
            "score":      env.points,
            "best_score": best_score,
            "speed":      env.game_speed,
            "fps":        env.clock.get_fps(),
            "reward":     genome_list[hud_i].fitness,
            "action_name": ACTION_NAMES[actions[hud_i]],

            "alive":          len(alive_indices),
            "generation":     generation,
            "fitness":        genome_list[hud_i].fitness,
            "genome":         genome_list[hud_i],
            "neural_outputs": outputs_all[hud_i],

            "state":       states[hud_i],
            "sensor_data": info.get("sensor_data"),
        }

        env.render(hud=hud, hud_data=hud_data)
        env.tick(FPS)

    # ── end of generation ─────────────────────────────────────────────────────

    best_genome_this_gen = max(genome_list, key=lambda g: g.fitness)
    best_fit = best_genome_this_gen.fitness

    hud.push_episode(env.points, best_fit)

    # save once per generation, only if this generation produced an overall best
    if best_genome is None or best_fit > best_genome.fitness:
        best_genome = best_genome_this_gen
        save_best_genome(best_genome)

    print(
        f"Gen {generation:3d} | "
        f"Score: {env.points:5d} | "
        f"Best: {best_score:5d} | "
        f"Max fitness: {best_fit:.2f}"
    )

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_best_genome(genome):

    os.makedirs("models", exist_ok=True)

    with open(BEST_GENOME_PATH, "wb") as f:

        pickle.dump(genome, f)

    print(
        f"[NEAT] Best genome saved: "
        f"{BEST_GENOME_PATH}"
    )

# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_checkpoint():
    """Retorna o caminho do checkpoint mais recente ou None."""
    prefix = "models/neat_checkpoint_"
    files  = glob.glob(f"{prefix}*")
    if not files:
        return None
    def gen_num(f):
        try:
            return int(os.path.basename(f).replace("neat_checkpoint_", ""))
        except ValueError:
            return -1
    return max(files, key=gen_num)


def run(resume=False):

    global generation

    config = neat.Config(

        neat.DefaultGenome,

        neat.DefaultReproduction,

        neat.DefaultSpeciesSet,

        neat.DefaultStagnation,

        CONFIG_PATH
    )

    # ── população: nova ou restaurada ──────────────────────────────────────────

    if resume:
        checkpoint = find_latest_checkpoint()
        if checkpoint:
            print(f"\n[NEAT] Continuando de: {checkpoint}\n")
            population = neat.Checkpointer.restore_checkpoint(checkpoint)
            generation = population.generation   # sincroniza contador de exibição
        else:
            print("\n[NEAT] Nenhum checkpoint encontrado — iniciando do zero.\n")
            population = neat.Population(config)
    else:
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(
        neat.Checkpointer(
            generation_interval=10,
            filename_prefix="models/neat_checkpoint_"
        )
    )

    os.makedirs("models", exist_ok=True)

    print("\n=== NEAT TRAINING STARTED ===\n")
    print("ESC / fechar janela → salva e volta ao menu\n")

    try:
        winner = population.run(eval_genomes, MAX_GENERATIONS)
        save_best_genome(winner)
        print("\nTreinamento concluido.\n")

    except TrainingAborted:
        print("\nTreinamento interrompido pelo usuario.\n")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    run()