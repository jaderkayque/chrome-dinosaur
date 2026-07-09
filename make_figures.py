# make_figures.py — generates real figures for the paper from saved data.
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import sys
sys.path.insert(0, ".")
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import neat

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 200,
})

OUT = "figs"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# FIG 1: NEAT fitness / complexity progression across checkpoints
# ---------------------------------------------------------------------------
def neat_progression():
    files = sorted(glob.glob("models/neat_checkpoint_*"),
                   key=lambda f: int(f.split("_")[-1]))
    gens, best, mean, nodes, conns = [], [], [], [], []
    for f in files:
        pop = neat.Checkpointer.restore_checkpoint(f)
        fits = [g.fitness for g in pop.population.values() if g.fitness is not None]
        if not fits:
            continue
        gens.append(pop.generation)
        best.append(max(fits))
        mean.append(float(np.mean(fits)))
        nodes.append(float(np.mean([len(g.nodes) for g in pop.population.values()])))
        conns.append(float(np.mean([sum(1 for c in g.connections.values() if c.enabled)
                                     for g in pop.population.values()])))

    fig, ax1 = plt.subplots(figsize=(3.4, 2.4))
    ax1.plot(gens, best, "o-", color="#1f77b4", label="best fitness", lw=1.4, ms=3)
    ax1.plot(gens, mean, "s--", color="#7aa6c2", label="mean fitness", lw=1.0, ms=2.5)
    ax1.set_xlabel("Geração")
    ax1.set_ylabel("Fitness")
    ax1.legend(loc="upper left", fontsize=7, framealpha=0.9)

    ax2 = ax1.twinx()
    ax2.plot(gens, conns, "^:", color="#d62728", label="conexões (média)", lw=1.0, ms=3)
    ax2.set_ylabel("Conexões ativas (média)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(False)

    fig.tight_layout()
    fig.savefig(f"{OUT}/neat_progression.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/neat_progression.png", bbox_inches="tight")
    print("wrote neat_progression")


# ---------------------------------------------------------------------------
# FIG 2: score distributions from the evaluation harness
# ---------------------------------------------------------------------------
def score_distributions():
    if not (os.path.exists("eval_dqn_scores.npy") and os.path.exists("eval_neat_scores.npy")):
        print("eval scores not found, skipping distributions")
        return
    dqn = np.load("eval_dqn_scores.npy")
    neat_s = np.load("eval_neat_scores.npy")

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    bp = ax.boxplot([dqn, neat_s], labels=["D3QN", "NEAT"],
                    patch_artist=True, widths=0.5, showfliers=True)
    for patch, c in zip(bp["boxes"], ["#1f77b4", "#2ca02c"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_ylabel("Pontuação (frames sobrevividos)")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(f"{OUT}/score_box.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/score_box.png", bbox_inches="tight")
    print("wrote score_box")


# ---------------------------------------------------------------------------
# FIG 3: evolved champion topology
# ---------------------------------------------------------------------------
def champion_topology():
    import pickle
    with open("neat_ai/models/neat_best.pkl", "rb") as f:
        g = pickle.load(f)

    labels = {-1: "dist_x", -2: "width", -3: "height", -4: "speed",
              -5: "dino_y", -6: "vel_y", -7: "is_bird", -8: "gap",
              -9: "next_type", 0: "RUN", 1: "JUMP", 2: "DUCK"}

    used_in = sorted({s for (s, d), c in g.connections.items() if c.enabled and s < 0},
                     reverse=True)
    hidden = sorted(k for k in g.nodes if k not in (0, 1, 2))
    outs = [0, 1, 2]

    pos = {}
    for i, k in enumerate(used_in):
        pos[k] = (0.0, 1.0 - i / max(len(used_in) - 1, 1))
    for i, k in enumerate(hidden):
        pos[k] = (0.5, 0.5)
    for i, k in enumerate(outs):
        pos[k] = (1.0, 1.0 - i / 2.0)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for (s, d), c in g.connections.items():
        if not c.enabled or s not in pos or d not in pos:
            continue
        col = "#2ca02c" if c.weight >= 0 else "#d62728"
        lw = 0.5 + min(abs(c.weight), 7) / 2.0
        ax.annotate("", xy=pos[d], xytext=pos[s],
                    arrowprops=dict(arrowstyle="->", color=col, lw=lw, alpha=0.8))

    for k, (x, y) in pos.items():
        if k < 0:
            color = "#aed6f1"
        elif k in outs:
            color = "#f9e79f"
        else:
            color = "#d2b4de"
        ax.scatter([x], [y], s=420, color=color, edgecolors="black", zorder=3)
        ax.text(x, y, labels.get(k, str(k)), ha="center", va="center",
                fontsize=6, zorder=4)

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.15, 1.15)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/champion_topology.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/champion_topology.png", bbox_inches="tight")
    print("wrote champion_topology")


if __name__ == "__main__":
    neat_progression()
    champion_topology()
    score_distributions()
